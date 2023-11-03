#include "golly/Analysis/PolyhedralBuilder.h"
#include "golly/Analysis/ConditionalDominanceAnalysis.h"
#include "golly/Analysis/CudaParameterDetection.h"
#include "golly/Analysis/PscopDetector.h"
#include "golly/Analysis/SccOrdering.h"
#include "golly/Analysis/StatementDetection.h"
#include "golly/Support/ConditionalVisitor.h"
#include "golly/Support/isl_llvm.h"

#include <fmt/format.h>

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/Support/FormatVariadic.h>

namespace golly {

using namespace islpp;

struct ScevAffinator
    : llvm::SCEVVisitor<ScevAffinator, llvm::Optional<pw_aff>> {
  using Base = llvm::SCEVVisitor<ScevAffinator, llvm::Optional<pw_aff>>;
  using RetVal = llvm::Optional<pw_aff>;
  using Combinator = pw_aff (*)(pw_aff, pw_aff);

  llvm::ScalarEvolution &se;
  const AffineContext &context;

  space sp;

  RetVal visitConstant(const llvm::SCEVConstant *cint) {
    return sp.constant<pw_aff>(cint->getAPInt().getSExtValue());
  }

  RetVal visitAddExpr(const llvm::SCEVAddExpr *S) {
    return mergeNary(S, [](pw_aff lhs, pw_aff rhs) { return lhs + rhs; });
  }

  RetVal visitMulExpr(const llvm::SCEVMulExpr *S) {
    return mergeNary(S, [](pw_aff lhs, pw_aff rhs) { return lhs * rhs; });
  }

  RetVal visitPtrToIntExpr(const llvm::SCEVPtrToIntExpr *S) {
    return visit(S->getOperand());
  }
  RetVal visitTruncateExpr(const llvm::SCEVTruncateExpr *S) {
    return visit(S->getOperand());
  }
  RetVal visitZeroExtendExpr(const llvm::SCEVZeroExtendExpr *S) {
    return visit(S->getOperand());
  }
  RetVal visitSignExtendExpr(const llvm::SCEVSignExtendExpr *S) {
    return visit(S->getOperand());
  }
  RetVal visitSMaxExpr(const llvm::SCEVSMaxExpr *S) {
    return mergeNary(S, static_cast<Combinator>(max));
  }
  RetVal visitUMaxExpr(const llvm::SCEVUMaxExpr *S) {
    return mergeNary(S, static_cast<Combinator>(max));
  }
  RetVal visitSMinExpr(const llvm::SCEVSMinExpr *S) {
    return mergeNary(S, static_cast<Combinator>(min));
  }
  RetVal visitUMinExpr(const llvm::SCEVUMinExpr *S) {
    return mergeNary(S, static_cast<Combinator>(min));
  }

  RetVal visitUDivExpr(const llvm::SCEVUDivExpr *S) {
    // unsigned division
    const auto dividend = S->getLHS();
    const auto divisor = S->getRHS();

    // divisor must be const to be affine
    return visitDivision(dividend, divisor, S);
  }
  RetVal visitSDivInstruction(llvm::Instruction *SDiv, const llvm::SCEV *Expr) {
    assert(SDiv->getOpcode() == llvm::Instruction::SDiv &&
           "Assumed SDiv instruction!");

    auto *Dividend = se.getSCEV(SDiv->getOperand(0));
    auto *Divisor = se.getSCEV(SDiv->getOperand(1));
    return visitDivision(Dividend, Divisor, Expr, SDiv);
  }
  RetVal visitDivision(const llvm::SCEV *dividend, const llvm::SCEV *divisor,
                       const llvm::SCEV *S, llvm::Instruction *inst = nullptr) {
    // todo
    return llvm::None;
  }

  RetVal visitAddRecExpr(const llvm::SCEVAddRecExpr *S) {
    const auto start = S->getStart();
    const auto step = S->getStepRecurrence(se);
    // llvm::dbgs() << *S << "\n";

    if (start->isZero()) {
      // todo
      auto step = visit(S->getOperand(1));
      if (!step)
        return llvm::None;
      // loop MUST exist
      auto indvar = S->getLoop()->getInductionVariable(se);

      if (!indvar)
        return llvm::None;

      // get loop index in context
      auto pos = context.getIVarIndex(indvar);
      assert(pos >= 0);

      // llvm::dbgs() << space << "\n";
      auto loop_expr = ISLPP_CHECK(sp.coeff<pw_aff>(dim::in, pos, 1));
      // llvm::dbgs() << loop_expr << "\n";
      // llvm::dbgs() << *step << "\n";
      return ISLPP_CHECK(loop_expr * *step);
    }

    auto zero_start = se.getAddRecExpr(se.getConstant(start->getType(), 0),
                                       S->getStepRecurrence(se), S->getLoop(),
                                       S->getNoWrapFlags());

    auto res_expr = visit(zero_start);
    auto start_expr = visit(start);

    if (res_expr && start_expr)
      return ISLPP_CHECK(*res_expr + *start_expr);
    else
      return llvm::None;
  }

  RetVal visitSequentialUMinExpr(const llvm::SCEVSequentialUMinExpr *S) {
    // todo
    return llvm::None;
  }
  RetVal visitUnknown(const llvm::SCEVUnknown *S) {
    const auto value = S->getValue();

    if (auto instr = llvm::dyn_cast<llvm::Instruction>(S->getValue())) {
      switch (instr->getOpcode()) {
      case llvm::BinaryOperator::SRem:
        return visitSRemInstruction(instr);
      default:
        break;
      }
    }

    if (auto itr = context.constants.find(value);
        itr != context.constants.end())
      return ISLPP_CHECK(sp.constant<pw_aff>(itr->second));

    if (context.parameters.contains(value)) {
      auto name = value->getName();
      auto param_sp = add_param(sp, name);
      return ISLPP_CHECK(param_sp.coeff<pw_aff>(
          dim::param, dims(param_sp, dim::param) - 1, 1));
    }

    if (int pos = context.getIVarIndex(value); pos != -1)
      return ISLPP_CHECK(sp.coeff<pw_aff>(dim::in, pos, 1));

    return llvm::None;
  }
  RetVal visitSRemInstruction(llvm::Instruction *instr) {
    auto lhs = visit(se.getSCEV(instr->getOperand(0)));
    auto rhs = visit(se.getSCEV(instr->getOperand(1)));
    if (lhs && rhs)
      return ISLPP_CHECK(*lhs % *rhs);
    else
      return llvm::None;
  }

  RetVal visitCouldNotCompute(const llvm::SCEVCouldNotCompute *S) {
    // todo
    return llvm::None;
  }

  RetVal mergeNary(const llvm::SCEVNAryExpr *S,
                   std::invocable<pw_aff, pw_aff> auto &&fn) {
    auto val = visit(S->getOperand(0));

    if (val) {
      for (int i = 1; i < S->getNumOperands(); ++i) {
        auto inVal = visit(S->getOperand(i));
        if (inVal)
          val = ISLPP_CHECK(fn(std::move(*val), std::move(*inVal)));
        else
          return llvm::None;
      }
    }

    return val;
  }

  using Base::visit;
  RetVal visit(llvm::Value *val) { return visit(se.getSCEV(val)); }
};

struct ConditionalAffinator
    : public ConditionalVisitor<ConditionalAffinator, set> {

  ScevAffinator &affinator;
  ConditionalAffinator(ScevAffinator &aff) : affinator{aff} {}

  set visitAnd(llvm::Instruction &and_inst) override {
    auto lhs = llvm::dyn_cast<llvm::Instruction>(and_inst.getOperand(0));
    auto rhs = llvm::dyn_cast<llvm::Instruction>(and_inst.getOperand(1));
    assert(lhs);
    assert(rhs);
    return ISLPP_CHECK(visit(*lhs) * visit(*rhs));
  };

  set visitOr(llvm::Instruction &or_inst) override {
    auto lhs = llvm::dyn_cast<llvm::Instruction>(or_inst.getOperand(0));
    auto rhs = llvm::dyn_cast<llvm::Instruction>(or_inst.getOperand(1));
    assert(lhs);
    assert(rhs);
    return ISLPP_CHECK(visit(*lhs) + visit(*rhs));
  };

  set visitSelectInst(llvm::SelectInst &select) {
    auto selector = visitValue(select.getOperand(0));
    auto true_branch = visitValue(select.getOperand(1));
    auto false_branch = visitValue(select.getOperand(2));

    return selector * true_branch;
  }

  set visitICmpInst(llvm::ICmpInst &icmp) override {
    auto lhs = icmp.getOperand(0);
    auto rhs = icmp.getOperand(1);
    assert(lhs);
    assert(rhs);

    auto lhscev = affinator.visit(lhs);
    auto rhscev = affinator.visit(rhs);

    if (lhscev && rhscev) {
      ISLPP_CHECK(*lhscev);
      ISLPP_CHECK(*rhscev);
      switch (icmp.getPredicate()) {
      case llvm::ICmpInst::Predicate::ICMP_EQ:
        return ISLPP_CHECK(eq_set(*lhscev, *rhscev));
      case llvm::ICmpInst::Predicate::ICMP_NE:
        return ISLPP_CHECK(ne_set(*lhscev, *rhscev));
      case llvm::ICmpInst::Predicate::ICMP_UGE:
      case llvm::ICmpInst::Predicate::ICMP_SGE:
        return ISLPP_CHECK(ge_set(*lhscev, *rhscev));
      case llvm::ICmpInst::Predicate::ICMP_UGT:
      case llvm::ICmpInst::Predicate::ICMP_SGT:
        return ISLPP_CHECK(gt_set(*lhscev, *rhscev));
      case llvm::ICmpInst::Predicate::ICMP_ULE:
      case llvm::ICmpInst::Predicate::ICMP_SLE:
        return ISLPP_CHECK(le_set(*lhscev, *rhscev));
      case llvm::ICmpInst::Predicate::ICMP_ULT:
      case llvm::ICmpInst::Predicate::ICMP_SLT:
        return ISLPP_CHECK(lt_set(*lhscev, *rhscev));
        break;
      default:
        break;
      }
    }

    return nullSet();
  }

  set visitInstruction(llvm::Instruction &) { return nullSet(); };

  set visitValue(llvm::Value *val) {
    if (auto instr = llvm::dyn_cast<llvm::Instruction>(val))
      return visit(instr);

    if (auto constant = llvm::dyn_cast<llvm::Constant>(val)) {
      assert(constant->getType()->getTypeID() == llvm::Type::IntegerTyID);
      auto val = constant->getUniqueInteger();

      if (val == 1)
        return ISLPP_CHECK(set{affinator.sp.universe<set>()});
      else
        return nullSet();
    }

    return nullSet();
  };

  set nullSet() { return ISLPP_CHECK(affinator.sp.empty<set>()); }
};

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

  struct ThreadExpressions {
    map tau2cta;
    map tau2thread;
    map thread2warpTuple;
    map warpTuple2warp;
    map warpTuple2lane;
  };

  ThreadExpressions createThreadExpressions() {
    const auto &global = detection.getGlobalContext();

    // get space
    auto sp = get_space(spacify(global, se));

    // create cta getter
    auto [tau2cta, tau2thread] = [&]() -> std::pair<multi_aff, multi_aff> {
      auto cta_expr = null_tuple(sp);
      auto thd_expr = null_tuple(sp);

      int index = 0;
      for (auto &iv : global.induction_vars) {
        if (iv.kind == InstantiationVariable::Kind::Block)
          cta_expr = ISLPP_CHECK(flat_range_product(
              cta_expr, sp.coeff<multi_aff>(dim::in, index, 1)));

        if (iv.kind == InstantiationVariable::Kind::Thread)
          thd_expr = ISLPP_CHECK(flat_range_product(
              thd_expr, sp.coeff<multi_aff>(dim::in, index, 1)));

        ++index;
      }

      return {cta_expr, thd_expr};
    }();

    auto [warptuple_expr, warp_expr,
          lane_expr] = [&]() -> std::tuple<map, map, islpp ::map> {
      int index = 0;
      auto multiplier = 1;

      auto expr = sp.zero<pw_aff>();

      for (auto &iv : global.induction_vars) {
        if (iv.kind == InstantiationVariable::Kind::Thread) {
          // z * nx * ny + y * nx + x

          expr = expr + (sp.coeff<pw_aff>(dim::in, index, 1) *
                         sp.constant<pw_aff>(multiplier));

          if (iv.dim)
            multiplier = multiplier * cuda.getDimCounts().at(*iv.dim);
        }

        ++index;
      }

      auto warp_size = sp.constant<pw_aff>(32);

      auto warp_getter = expr / warp_size;
      auto lane_getter = expr % warp_size;

      auto warp_map = map(warp_getter);
      auto lane_map = map(lane_getter);

      auto warp_tuple = map{ISLPP_CHECK(flat_range_product(
          cast<multi_pw_aff>(warp_getter), cast<multi_pw_aff>(lane_getter)))};

      return {warp_tuple, apply_range(reverse(warp_tuple), warp_map),
              apply_range(reverse(warp_tuple), lane_map)};
    }();

    return ThreadExpressions{.tau2cta = map{tau2cta},
                             .tau2thread = map{tau2thread},
                             .thread2warpTuple = warptuple_expr,
                             .warpTuple2warp = warp_expr,
                             .warpTuple2lane = lane_expr};
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
        // llvm::dbgs() << "param name: " << param << "\n";
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

  union_map constructValidBarriers(const ThreadExpressions &thread_exprs,
                                   union_map domain, union_map thread_alloc,
                                   union_map temporal_schedule) {
    // [S->T] -> Stmt
    union_map beta{"{}"};

    union_map rev_alloc = reverse(thread_alloc);
    scc.traverse(
        [&](const llvm::BasicBlock *bb) {
          for (auto &stmt : stmts.iterateStatements(*bb)) {
            if (auto bar_stmt = stmt.as<golly::BarrierStatement>()) {
              union_set s{name(set{"{[]}"}, bar_stmt->getName())};
              auto stmt_instances = apply(s, domain);
              auto tid_to_insts = range_intersect(rev_alloc, stmt_instances);

              auto &barrier = bar_stmt->getBarrier();
              const auto active_threads = apply(stmt_instances, thread_alloc);

              if (is_empty(active_threads)) {
                llvm::outs()
                    << "warning: unreachable barrier\n"
                    << *bar_stmt->getDefiningInstruction().getDebugLoc().get()
                    << "\n";
                continue;
              }

              const auto thread_pairs =
                  universal(active_threads, active_threads) -
                  identity(active_threads);

              if (auto warp_bar =
                      std::get_if<golly::BarrierStatement::Warp>(&barrier)) {
                // collect all warps in this statement

                // convert the mask to a set
                warp_bar->mask;

                // collect all expected warps
              }

              if (auto block_bar =
                      std::get_if<golly::BarrierStatement::Block>(&barrier)) {
                // check for barrier divergence
                {
                  // collect all ctas in this statement
                  auto ctas =
                      apply(active_threads, union_map{thread_exprs.tau2cta});

                  // generate all expected threads
                }

                // otherwise, perform a simple operation to determine which
                // domains synchronize inst -> tid
                auto dmn_insts =
                    apply_range(domain_map(thread_pairs), tid_to_insts) +
                    apply_range(range_map(thread_pairs), tid_to_insts);

                // filter to same gids
                auto same_ctaid =
                    apply_range(union_map{thread_exprs.tau2cta},
                                reverse(union_map{thread_exprs.tau2cta}));

                beta = beta + domain_intersect(dmn_insts, wrap(same_ctaid));
              }

              if (auto global_bar =
                      std::get_if<golly::BarrierStatement::End>(&barrier)) {
                auto dmn_insts =
                    apply_range(domain_map(thread_pairs), tid_to_insts) +
                    apply_range(range_map(thread_pairs), tid_to_insts);
                beta = beta + coalesce(dmn_insts);
              }
            }
          }
        });
    return coalesce(beta);
  }

  union_map constructSynchronizationSchedule(const ThreadExpressions &thd_exprs,
                                             union_map domain,
                                             union_map thread_allocation,
                                             union_map temporal_schedule) {
    auto beta = constructValidBarriers(thd_exprs, domain, thread_allocation,
                                       temporal_schedule);

    auto tid_to_stmt_inst = reverse(thread_allocation);

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

    return sync_times;
  }

  std::pair<union_map, union_map> calculateAccessRelations(union_map domain) {
    union_map reads{"{}"};
    union_map writes{"{}"};

    scc.traverse([&](const llvm::BasicBlock *bb) {
      auto loop = li.getLoopFor(bb);
      auto &context = detection.getLoopInfo(loop)->context;
      auto sp = get_space(spacify(context, se));
      auto affinator = ScevAffinator{.se = se, .context = context, .sp = sp};

      for (auto &stmt : stmts.iterateStatements(*bb)) {
        if (auto mem_acc = stmt.as<golly::MemoryAccessStatement>()) {
          auto ptr = mem_acc->getPointer();
          auto ptr_name = ptr->getName();

          if (auto offset = mem_acc->getOffset()) {
            if (auto val = affinator.visit(const_cast<llvm::Value *>(offset))) {
              auto as_map = name(name(map{*val}, dim::out, ptr_name), dim::in,
                                 stmt.getName());
              if (mem_acc->getAccessType() ==
                  MemoryAccessStatement::Access::Read)
                reads = reads + union_map{as_map};
              else
                writes = writes + union_map{as_map};
            } else {
              // non-affine offset
              // ignore
            }
          } else {
            // there is no offset
            // treat it as 0
            auto zero = sp.zero<pw_aff>();
            auto as_map = name(name(map{zero}, dim::out, ptr_name), dim::in,
                               stmt.getName());
            if (mem_acc->getAccessType() == MemoryAccessStatement::Access::Read)
              reads = reads + union_map{as_map};
            else {
              assert(mem_acc->getAccessType() ==
                     MemoryAccessStatement::Access::Write);
              writes = writes + union_map{as_map};
            }
          }
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
  const auto sync_schedule = builder.constructSynchronizationSchedule(
      exprs, domain, thread_alloc, temporal_schedule);
  const auto [reads, writes] = builder.calculateAccessRelations(domain);

  return Pscop{
      .instantiation_domain = domain,
      .thread_allocation = thread_alloc,
      .temporal_schedule = temporal_schedule,
      .sync_schedule = sync_schedule,
      .write_access_relation = writes,
      .read_access_relation = reads,
  };
}
} // namespace golly