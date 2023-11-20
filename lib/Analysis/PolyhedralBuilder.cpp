#include "golly/Analysis/PolyhedralBuilder.h"
#include "PolyhedralBuilderSupport.h"
#include "golly/Analysis/ConditionalDominanceAnalysis.h"
#include "golly/Analysis/CudaParameterDetection.h"
#include "golly/Analysis/PscopDetector.h"
#include "golly/Analysis/SccOrdering.h"
#include "golly/Analysis/StatementDetection.h"
#include "golly/Support/ConditionalVisitor.h"
#include "golly/Support/isl_llvm.h"

#include <fmt/format.h>

#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/FormatVariadic.h>

namespace golly {

using namespace islpp;

pw_aff valuate(golly::InstantiationVariable::Expr expr,
               const AffineContext &context, llvm::ScalarEvolution &se,
               space sp) {
  if (auto val = std::get_if<llvm::Value *>(&expr)) {
    auto affinator = ScevAffinator{
        .se = se,
        .context = context,
        .sp = sp,
    };
    return *affinator.visit(se.getSCEV(*val));
  } else {
    return sp.constant<pw_aff>(std::get<int>(expr));
  }
}

// hack it
multi_aff null_tuple(space sp) {
  return multi_aff{isl_multi_aff_multi_val_on_space(
      sp.yield(), isl_multi_val_read_from_str(ctx(), "{[]}"))};
}

set spacify(const AffineContext &context, llvm::ScalarEvolution &se) {
  set s{"{ [] }"};
  s = add_dims(std::move(s), dim::set, context.induction_vars.size());

  auto sp = get_space(s);
  int i = 0;
  for (auto &iv : context.induction_vars) {
    auto lb = valuate(iv.lower_bound, context, se, sp);
    auto ub = valuate(iv.upper_bound, context, se, sp);
    auto identity = sp.coeff<pw_aff>(dim::in, i++, 1);
    auto set = le_set(lb, identity) * lt_set(identity, ub);
    s = ISLPP_CHECK(set * std::move(s));
  }

  return s;
}

set consolidate(llvm::Value *conditional, llvm::ScalarEvolution &se, space sp,
                const AffineContext &context) {
  ScevAffinator affinator{.se = se, .context = context, .sp = sp};
  // llvm::dbgs() << *conditional << "\n";
  if (auto instr = llvm::dyn_cast<llvm::Instruction>(conditional))
    return ISLPP_CHECK(ConditionalAffinator{affinator}.visit(*instr));
  if (auto constant = llvm::dyn_cast<llvm::Constant>(conditional))
    return ISLPP_CHECK(ConditionalAffinator{affinator}.visitValue(constant));

  return set{"{}"};
}

struct PolyhedralBuilder {
  llvm::Function &f;
  ConditionalDominanceAnalysis &cda;
  SccOrdering &scc;
  StatementDetection &stmts;
  CudaParameters &cuda;
  PscopDetection &detection;
  llvm::ScalarEvolution &se;
  llvm::LoopInfo &li;

  ThreadExpressions createThreadExpressions() {
    const auto &global = detection.getGlobalContext();

    // get space
    auto sp = get_space(spacify(global, se));

    // create cta getter
    auto [tau2cta, tau2thread,
          allthread] = [&]() -> std::tuple<multi_aff, multi_aff, set> {
      auto cta_expr = null_tuple(sp);
      auto thd_expr = null_tuple(sp);
      auto thd_set = set{"{ [] }"};

      int index = 0;
      for (auto &iv : global.induction_vars) {
        if (iv.kind == InstantiationVariable::Kind::Block)
          cta_expr = ISLPP_CHECK(flat_range_product(
              cta_expr, sp.coeff<multi_aff>(dim::in, index, 1)));

        if (iv.kind == InstantiationVariable::Kind::Thread) {
          thd_expr = ISLPP_CHECK(flat_range_product(
              thd_expr, sp.coeff<multi_aff>(dim::in, index, 1)));
          auto new_sp = get_space(set{"{ [i] }"});
          auto id = ISLPP_CHECK(new_sp.coeff<aff>(dim::in, 0, 1));
          auto lb = new_sp.zero<aff>();
          assert(std::get_if<int>(&iv.upper_bound) &&
                 "thread id upper bound should be fixed integer");
          auto ub = new_sp.constant<aff>(std::get<int>(iv.upper_bound));

          auto tid_set = le_set(lb, id) * lt_set(id, ub);
          thd_set = flat_cross(std::move(thd_set), std::move(tid_set));
        }
        ++index;
      }

      return {cta_expr, thd_expr, thd_set};
    }();

    auto [warptuple_expr, warp_expr,
          lane_expr] = [&]() -> std::tuple<map, map, islpp ::map> {
      int index = 0;
      auto multiplier = 1;

      auto flat_thread_expr = sp.zero<pw_aff>();

      for (auto &iv : global.induction_vars) {
        if (iv.kind == InstantiationVariable::Kind::Thread) {
          // z * nx * ny + y * nx + x

          flat_thread_expr =
              flat_thread_expr + (sp.coeff<pw_aff>(dim::in, index, 1) *
                                  sp.constant<pw_aff>(multiplier));

          if (iv.dim)
            multiplier = multiplier * cuda.getDimCounts().at(*iv.dim);
        }

        ++index;
      }

      auto warp_size = sp.constant<pw_aff>(32);

      auto warp_getter = flat_thread_expr / warp_size;
      auto lane_getter = flat_thread_expr % warp_size;

      auto warp_map = map(warp_getter);
      auto lane_map = map(lane_getter);

      auto warp_tuple = map{ISLPP_CHECK(flat_range_product(
          cast<multi_pw_aff>(warp_getter), cast<multi_pw_aff>(lane_getter)))};

      return {warp_tuple, apply_range(reverse(warp_tuple), warp_map),
              apply_range(reverse(warp_tuple), lane_map)};
    }();

    return ThreadExpressions{
        .tau2cta = map{tau2cta},
        .tau2thread = map{tau2thread},
        .tau2warpTuple = warptuple_expr,
        .warpTuple2warp = warp_expr,
        .warpTuple2lane = lane_expr,
        .globalThreadSet = allthread,
    };
  }

  union_map constructDomain() {
    llvm::DenseMap<const llvm::BasicBlock *, set> domains;

    scc.traverse([&](const llvm::BasicBlock *bb) {
      auto loop = li.getLoopFor(bb);
      auto loop_info = detection.getLoopInfo(loop);
      auto domain = spacify(loop_info->context, se);

      // llvm::dbgs() << bb->getName() << domain << "\n";
      domains.insert({bb, std::move(domain)});
    });

    for (auto br : cda.getBranches()) {
      auto loop = li.getLoopFor(br->getParent());
      auto loop_info = detection.getLoopInfo(loop);
      auto space = get_space(domains[br->getParent()]);
      auto br_dims = dims(space, dim::set);
      if (auto cond = detection.getBranchInfo(br)) {
        auto true_set = ISLPP_CHECK(
            consolidate(br->getCondition(), se, space, loop_info->context));
        for (auto &bb : cda.getTrueBranch(br)) {
          auto &dom = domains[bb];
          auto diff = dims(dom, dim::set) - br_dims;
          dom = ISLPP_CHECK(add_dims(true_set, dim::set, diff) * dom);
        }

        for (auto &bb : cda.getFalseBranch(br)) {
          auto &dom = domains[bb];
          auto diff = dims(dom, dim::set) - br_dims;
          dom = ISLPP_CHECK(dom - add_dims(true_set, dim::set, diff));
        }
      } else {
        // non-affine branch, introduce a param to distinguish taken and not
        // taken
        static int counter = 0;
        auto param = add_param(space, llvm::formatv("b{0}", counter++).str());
        auto param_count = dims(param, dim::param);
        auto val = param.coeff<pw_aff>(dim::param, param_count - 1, 1);

        auto zero = param.constant<pw_aff>(0);
        auto one = param.constant<pw_aff>(1);

        auto true_set = eq_set(val, zero);
        auto false_set = eq_set(val, one);
        for (auto &bb : cda.getTrueBranch(br)) {
          auto &dom = domains[bb];
          auto diff = dims(dom, dim::set) - br_dims;
          dom = ISLPP_CHECK(add_dims(true_set, dim::set, diff) * dom);
        }

        for (auto &bb : cda.getFalseBranch(br)) {
          auto &dom = domains[bb];
          auto diff = dims(dom, dim::set) - br_dims;
          dom = ISLPP_CHECK(dom - add_dims(true_set, dim::set, diff));
        }
      }
    }

    // apply the domain to all sets
    union_map ret{"{}"};

    for (auto &[bb, domain] : domains) {
      for (auto &stmt : stmts.iterateStatements(*bb)) {
        auto s = name(set{"{[]}"}, stmt.getName());
        auto instances = name(domain, stmt.getName());
        ret = ret + union_map{unwrap(cross(s, instances))};
      }
    }

    return ret;
  }

  union_map constructDistribution(union_map domain) {
    const auto thread_dims = detection.getGlobalContext().induction_vars.size();
    // we need to project out the thread dims for all statements
    auto instances = range(domain);

    union_map ret;

    for_each(instances, [&ret, thread_dims](set set) {
      auto sp = get_space(set);
      std::vector<aff> affs;
      for (int i = 0; i < thread_dims; ++i)
        affs.emplace_back(sp.coeff<aff>(dim::in, i, 1));

      auto as_map = map{flat_range_product(affs)};
      ret = ret + union_map{domain_intersect(as_map, set)};
    });

    return ret;
  }

  union_map constructTemporalSchedule(union_map domain) {
    struct LoopTime {
      multi_aff prefix_expr;
      space sp;
      int count = 0;
    };

    llvm::DenseMap<llvm::Loop *, LoopTime> times;
    auto sp = get_space(spacify(detection.getGlobalContext(), se));

    // for time, we want to construct a null prefix time that we can build up
    times[nullptr] =
        LoopTime{.prefix_expr = null_tuple(sp), .sp = sp, .count = 0};

    auto get_parent = [this](llvm::Loop *loop) -> llvm::Loop * {
      loop = loop->getParentLoop();
      while (loop) {
        if (detection.getLoopInfo(loop)->is_affine)
          return loop;
        loop = loop->getParentLoop();
      }
      return nullptr;
    };

    const int max_depth = [&]() {
      auto calcDepth = [&get_parent](llvm::Loop *loop) -> int {
        int depth = 0;
        while (loop) {
          depth++;
          loop = get_parent(loop);
        }

        return depth;
      };

      int max_depth = 0;
      for (auto &loop : li.getLoopsInPreorder())
        max_depth = std::max(calcDepth(loop), max_depth);

      // 1 initial coutner + loops * (1 iteration var + 1 counter var)
      return 1 + 2 * max_depth;
    }();

    union_map ret{"{}"};
    scc.traverse([&](const llvm::BasicBlock *bb) {
      auto loop = li.getLoopFor(bb);
      auto loop_context = detection.getLoopInfo(loop);
      auto itr = times.find(loop_context->affine_loop);

      if (itr == times.end()) {
        // new affine loop
        auto &metadata = times[get_parent(loop)];
        auto my_count = metadata.count++;
        auto my_sp = add_dims(metadata.sp, dim::set, 1);
        auto my_expr = ([&]() {
          auto v = ISLPP_CHECK(add_dims(metadata.prefix_expr, dim::in, 1));
          auto c = ISLPP_CHECK(my_sp.constant<multi_aff>(my_count));
          auto vc = ISLPP_CHECK(flat_range_product(std::move(v), std::move(c)));
          auto p = ISLPP_CHECK(
              my_sp.coeff<multi_aff>(dim::in, dims(my_sp, dim::set) - 1, 1));
          return ISLPP_CHECK(flat_range_product(std::move(vc), std::move(p)));
        })();

        times[loop_context->affine_loop] =
            LoopTime{.prefix_expr = my_expr, .sp = my_sp, .count = 0};
        itr = times.find(loop_context->affine_loop);
      }

      // loop already exists, increment the counter
      auto &metadata = itr->second;
      auto sp = metadata.sp;
      for (auto &elem : stmts.iterateStatements(*bb)) {
        auto my_count = metadata.count++;
        auto my_expr = ISLPP_CHECK(flat_range_product(
            metadata.prefix_expr, sp.constant<multi_aff>(my_count)));

        while (dims(my_expr, dim::out) < max_depth)
          my_expr = flat_range_product(my_expr, sp.zero<multi_aff>());

        auto my_map = map{name(my_expr, dim::in, elem.getName())};
        ret = ret + ISLPP_CHECK(union_map{my_map});
      }
    });

    ret = domain_intersect(ret, range(domain));

    return ret;
  }

  std::pair<union_map, ErrorList>
  constructValidBarriers(const ThreadExpressions &thread_exprs,
                         union_map stmt_domain, union_map thread_alloc,
                         union_map temporal_schedule) {
    // [S->T] -> Stmt
    union_map beta{"{}"};
    ErrorList errs;

    scc.traverse([&](const llvm::BasicBlock *bb) {
      for (auto &stmt : stmts.iterateStatements(*bb)) {
        if (auto bar_stmt = stmt.as<golly::BarrierStatement>()) {
          // llvm::dbgs() << "HEY\n";
          // use time as a measure of barrier instances
          // time -> tau

          union_set s{name(set{"{[]}"}, bar_stmt->getName())};
          auto stmt_instances = apply(s, stmt_domain);

          // llvm::dbgs() << "insts: " << stmt_instances << "\n";

          if (is_empty(stmt_instances)) {
            llvm::outs()
                << "warning: unreachable barrier\n"
                << *bar_stmt->getDefiningInstruction().getDebugLoc().get()
                << "\n";
            continue;
          }
          auto time_map =
              apply_range(identity(stmt_instances), temporal_schedule);

          const auto tau_map =
              apply_range(identity(stmt_instances), thread_alloc);

          // b -> tau
          auto beta_tau = apply_range(reverse(time_map), tau_map);
          auto same_tau = wrap(identity(range(tau_map)));

          // b -> [tau -> tau]
          auto beta_tau_pairs =
              range_subtract(range_product(beta_tau, beta_tau),
                             wrap(identity(range(beta_tau))));

          auto &barrier = bar_stmt->getBarrier();
          if (auto warp_bar =
                  std::get_if<golly::BarrierStatement::Warp>(&barrier)) {
            // collect all warps in this statement

            auto tau2warp = ISLPP_CHECK(apply_range(
                thread_exprs.tau2warpTuple, thread_exprs.warpTuple2warp));
            auto tau2lane = ISLPP_CHECK(apply_range(
                thread_exprs.tau2warpTuple, thread_exprs.warpTuple2lane));

            auto beta_warps = range_product(
                apply_range(beta_tau, union_map{thread_exprs.tau2cta}),
                apply_range(beta_tau, union_map{tau2warp}));
            auto beta_lanes = apply_range(beta_tau, union_map{tau2lane});

            // convert the mask to a set
            auto s = MaskAffinator{}.visitValue(warp_bar->mask);

            auto waiting_beta_lanes =
                universal(domain(beta_warps), union_set(s));
            // generate all expected warps
            auto waiting_warplanes =
                range_product(beta_warps, waiting_beta_lanes);
            auto active_warplanes =
                coalesce(range_product(beta_warps, beta_lanes));

            // llvm::dbgs() << "waiting: " << waiting_warplanes << "\n";
            // llvm::dbgs() << "active: " << active_warplanes << "\n";

            if ((waiting_warplanes > active_warplanes)) {
              errs.emplace_back(BarrierDivergence{
                  .barrier = bar_stmt,
                  .level = Level::Warp,
              });
              continue;
            }

            // synchronize the waiting warplanes
            // right now, waiting_warplanes is:
            // beta -> [ [cta -> w] -> l ]
            // we need to isolate [cta->w], and also recover [w->l] to recover
            // the thread

            const auto inst_to_cta =
                apply_range(tau_map, union_map{thread_exprs.tau2cta});
            const auto same_cta =
                apply_range(inst_to_cta, reverse(inst_to_cta));

            const auto inst_to_warp = chain_apply_range(
                tau_map, union_map{thread_exprs.tau2warpTuple},
                union_map{thread_exprs.warpTuple2warp});
            const auto same_warp =
                apply_range(inst_to_warp, reverse(inst_to_warp));

            const auto same_time = apply_range(time_map, reverse(time_map));

            const auto in_waiting_lane = range_intersect(
                chain_apply_range(tau_map,
                                  union_map{thread_exprs.tau2warpTuple},
                                  union_map{thread_exprs.warpTuple2lane}),
                union_set{s});

            auto syncing_stmts =
                same_cta * same_warp * same_time *
                universal(domain(in_waiting_lane), domain(in_waiting_lane));

            auto inst_to_S = apply_range(domain_map(syncing_stmts), tau_map);
            auto inst_to_T = apply_range(range_map(syncing_stmts), tau_map);

            auto tmp =
                domain_subtract(domain_factor_range(reverse(
                                    range_product(inst_to_S, inst_to_T))),
                                same_tau);
            beta = beta +
                   domain_subtract(domain_factor_range(reverse(
                                       range_product(inst_to_S, inst_to_T))),
                                   same_tau);
          }

          if (auto block_bar =
                  std::get_if<golly::BarrierStatement::Block>(&barrier)) {
            // check for barrier divergence

            // collect all ctas in this statement
            auto ctas = apply_range(beta_tau, union_map{thread_exprs.tau2cta});

            // generate all expected threads
            {
              auto waiting_taus = flat_range_product(
                  ctas, universal(domain(ctas),
                                  union_set{thread_exprs.globalThreadSet}));
              // llvm::dbgs() << "\nwaiting: " << waiting_taus << "\n";
              // llvm::dbgs() << "active: " << beta_tau << "\n";
              if ((waiting_taus != beta_tau)) {
                errs.emplace_back(BarrierDivergence{
                    .barrier = bar_stmt,
                    .level = Level::Block,
                });
                continue;
              }
            }

            // map statement instances to ctas
            const auto inst_to_cta =
                apply_range(tau_map, union_map{thread_exprs.tau2cta});

            const auto same_cta =
                apply_range(inst_to_cta, reverse(inst_to_cta));
            const auto same_time = apply_range(time_map, reverse(time_map));

            // SInst -> TInst
            auto syncing_stmts = same_cta * same_time;

            auto inst_to_S = apply_range(domain_map(syncing_stmts), tau_map);
            auto inst_to_T = apply_range(range_map(syncing_stmts), tau_map);

            beta = beta +
                   domain_subtract(domain_factor_range(reverse(
                                       range_product(inst_to_S, inst_to_T))),
                                   same_tau);
          }

          if (auto global_bar =
                  std::get_if<golly::BarrierStatement::End>(&barrier)) {
            auto tid_to_insts =
                range_intersect(reverse(thread_alloc), stmt_instances);
            const auto active_threads = range(thread_alloc);
            const auto thread_pairs =
                universal(active_threads, active_threads) -
                identity(active_threads);
            auto dmn_insts =
                apply_range(domain_map(thread_pairs), tid_to_insts) +
                apply_range(range_map(thread_pairs), tid_to_insts);
            beta = beta + dmn_insts;
          }
        }
      }
    });

    // llvm::dbgs() << "beta: " << beta << "\n";
    return {coalesce(beta), errs};
  }

  std::pair<union_map, ErrorList> constructSynchronizationSchedule(
      const ThreadExpressions &thd_exprs, union_map domain,
      union_map thread_allocation, union_map temporal_schedule) {
    auto [beta, errs] = constructValidBarriers(
        thd_exprs, domain, thread_allocation, temporal_schedule);

    // early terminate if barrier divergence is detected
    if (errs)
      return {union_map{"{}"}, errs};

    auto tid_to_stmt_inst =
        reverse(domain_intersect(thread_allocation, range(domain)));

    // enumerate all thread pairs
    auto threads = range(thread_allocation);
    auto thread_pairs = universal(threads, threads) - identity(threads);

    // [S -> T] -> StmtInsts of S and T
    auto dmn_insts =
        apply_range(domain_map(thread_pairs),
                    tid_to_stmt_inst) + // [ S -> T ] -> StmtInsts(S)
        apply_range(range_map(thread_pairs),
                    tid_to_stmt_inst); // [ S -> T ] -> StmtInsts(T)

    // [[S -> T] -> StmtInst] -> Time
    auto dmn_timings = apply_range(range_map(dmn_insts), temporal_schedule);

    // [[S -> T] -> StmtInst] -> Time
    auto sync_timings = apply_range(range_map(beta), temporal_schedule);

    // [[S->T] -> StmtInst] -> [[S->T] -> SyncInst]
    auto barrs_over_stmts = ([&]() {
      //  [[S -> T] -> StmtInst] -> [[S -> T] -> SyncInst]
      // but we have mismatched S->T
      auto bars_lex_stmts = std::move(dmn_timings) <<= std::move(sync_timings);

      // first, we zip to obtain: [[S->T] -> [S->T]] -> [StmtInst -> SyncInst]
      auto zipped = zip(std::move(bars_lex_stmts));

      // then we filter to [[S->T] == [S->T]] -> [StmtInst -> SyncInst]
      auto filtered = domain_intersect(std::move(zipped),
                                       wrap(identity(wrap(thread_pairs))));
      // then we unzip to retrieve the original
      // [[S->T] -> StmtInst] -> [[S->T] -> SyncInst]
      return zip(std::move(filtered));
    })();

    // [[S->T] -> StmtInst] -> SyncTime
    auto sync_times = lexmin(
        apply_range(range_factor_range(barrs_over_stmts), temporal_schedule));

    return {sync_times, errs};
  }

  std::pair<union_map, union_map> calculateAccessRelations(union_map domain) {
    union_map reads{"{}"};
    union_map writes{"{}"};

    auto get_cta = [&](space sp) -> islpp::multi_aff {
      multi_aff expr = null_tuple(sp);
      int index = 0;
      for (auto &&iv : detection.getGlobalContext().induction_vars) {
        if (iv.kind == InstantiationVariable::Kind::Block)
          expr =
              flat_range_product(expr, sp.coeff<multi_aff>(dim::in, index, 1));
        index++;
      }
      return expr;
    };

    scc.traverse([&](const llvm::BasicBlock *bb) {
      auto loop = li.getLoopFor(bb);
      auto &context = detection.getLoopInfo(loop)->context;
      auto sp = get_space(spacify(context, se));
      auto affinator = PtrScevAffinator{
          .ptr =
              PtrAffinator{
                  .se = se,
              },
          .scev = ScevAffinator{.se = se, .context = context, .sp = sp}};

      for (auto &stmt : stmts.iterateStatements(*bb)) {
        if (auto mem_acc = stmt.as<golly::MemoryAccessStatement>()) {
          auto val = mem_acc->getAddressOperand();
          auto [ptr, scev] =
              affinator.visit(se.getSCEV(const_cast<llvm::Value *>(val)));

          if (!(ptr && scev))
            continue;

          auto name_expr = [&, ptr_name = llvm::demangle(
                                   (*ptr)->getName().str())](auto expr) -> map {
            return name(name(map{expr}, dim::out, ptr_name), dim::in,
                        stmt.getName());
          };

          auto ptr_expr = map{cuda.isSharedMemoryPtr(*ptr)
                                  ? get_cta(sp)      // [cta, tid] -> [cta]
                                  : null_tuple(sp)}; // [cta, tid] -> []

          auto as_map = name_expr(flat_range_product(ptr_expr, map{*scev}));

          if (mem_acc->getAccessType() == MemoryAccessStatement::Access::Read)
            reads = reads + union_map{as_map};
          else
            writes = writes + union_map{as_map};
        }
      }
    });

    reads = domain_intersect(reads, range(domain));
    writes = domain_intersect(writes, range(domain));
    return {reads, writes};
  }
};

Pscop PolyhedralBuilderPass::run(llvm::Function &f,
                                 llvm::FunctionAnalysisManager &fam) {
  PolyhedralBuilder builder{
      .f = f,
      .cda = fam.getResult<golly::ConditionalDominanceAnalysisPass>(f),
      .scc = fam.getResult<golly::SccOrderingAnalysis>(f),
      .stmts = fam.getResult<golly::StatementDetectionPass>(f),
      .cuda = fam.getResult<golly::CudaParameterDetection>(f),
      .detection = fam.getResult<golly::PscopDetectionPass>(f),
      .se = fam.getResult<llvm::ScalarEvolutionAnalysis>(f),
      .li = fam.getResult<llvm::LoopAnalysis>(f),
  };

  const auto exprs = builder.createThreadExpressions();
  const auto domain = builder.constructDomain();
  const auto thread_alloc = builder.constructDistribution(domain);
  const auto temporal_schedule = builder.constructTemporalSchedule(domain);
  const auto [sync_schedule, errors] = builder.constructSynchronizationSchedule(
      exprs, domain, thread_alloc, temporal_schedule);
  const auto [reads, writes] = builder.calculateAccessRelations(domain);

  return Pscop{
      .instantiation_domain = domain,
      .thread_allocation = thread_alloc,
      .temporal_schedule = temporal_schedule,
      .sync_schedule = sync_schedule,
      .write_access_relation = writes,
      .read_access_relation = reads,

      .thread_expressions = exprs,
      .errors = errors,
  };
}
} // namespace golly