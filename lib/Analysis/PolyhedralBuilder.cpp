#include "golly/Analysis/PolyhedralBuilder.h"
#include "golly/Analysis/ConditionalDominanceAnalysis.h"
#include "golly/Analysis/PscopDetector.h"
#include "golly/Analysis/SccOrdering.h"
#include "golly/Analysis/StatementDetection.h"
#include "golly/Support/ConditionalVisitor.h"
#include "golly/Support/isl_llvm.h"

#include <fmt/format.h>

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>

namespace golly {

struct ScevAffinator
    : llvm::SCEVVisitor<ScevAffinator, llvm::Optional<islpp::pw_aff>> {
  using Base = llvm::SCEVVisitor<ScevAffinator, llvm::Optional<islpp::pw_aff>>;
  using RetVal = llvm::Optional<islpp::pw_aff>;
  using Combinator = islpp::pw_aff (*)(islpp::pw_aff, islpp::pw_aff);

  llvm::ScalarEvolution &se;
  const AffineContext &context;

  islpp::space space;

  RetVal visitConstant(const llvm::SCEVConstant *cint) {
    return space.constant<islpp::pw_aff>(cint->getAPInt().getSExtValue());
  }

  RetVal visitAddExpr(const llvm::SCEVAddExpr *S) {
    return mergeNary(
        S, [](islpp::pw_aff lhs, islpp::pw_aff rhs) { return lhs + rhs; });
  }

  RetVal visitMulExpr(const llvm::SCEVMulExpr *S) {
    return mergeNary(
        S, [](islpp::pw_aff lhs, islpp::pw_aff rhs) { return lhs * rhs; });
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
    return mergeNary(S, static_cast<Combinator>(islpp::max));
  }
  RetVal visitUMaxExpr(const llvm::SCEVUMaxExpr *S) {
    return mergeNary(S, static_cast<Combinator>(islpp::max));
  }
  RetVal visitSMinExpr(const llvm::SCEVSMinExpr *S) {
    return mergeNary(S, static_cast<Combinator>(islpp::min));
  }
  RetVal visitUMinExpr(const llvm::SCEVUMinExpr *S) {
    return mergeNary(S, static_cast<Combinator>(islpp::min));
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

    if (start->isZero()) {
      // todo
      auto step = visit(S->getOperand(1));
      if (!step)
        return llvm::None;
      // loop MUST exist
      auto indvar = S->getLoop()->getCanonicalInductionVariable();

      // get loop index in context
      auto pos = context.getIndexOfIVar(indvar);
      assert(pos >= 0);

      auto loop_expr =
          ISLPP_CHECK(space.coeff<islpp::pw_aff>(islpp::dim::in, pos, 1));
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
      return ISLPP_CHECK(space.constant<islpp::pw_aff>(itr->second));

    if (context.parameters.contains(value)) {
      auto name = value->getName();
      auto param_space = add_param(space, name);
      return ISLPP_CHECK(param_space.coeff<islpp::pw_aff>(
          islpp::dim::param, dims(param_space, islpp::dim::param) - 1, 1));
    }

    if (int pos = context.getIndexOfIVar(value); pos != -1)
      return ISLPP_CHECK(space.coeff<islpp::pw_aff>(islpp::dim::in, pos, 1));

    return llvm::None;
  }
  RetVal visitSRemInstruction(llvm::Instruction *instr) {
    auto lhs = visit(se.getSCEV(instr->getOperand(0)));
    auto rhs = visit(se.getSCEV(instr->getOperand(1)));
    if (lhs && rhs)
      return *lhs % *rhs;
    else
      return llvm::None;
  }

  RetVal visitCouldNotCompute(const llvm::SCEVCouldNotCompute *S) {
    // todo
    return llvm::None;
  }

  RetVal mergeNary(const llvm::SCEVNAryExpr *S,
                   std::invocable<islpp::pw_aff, islpp::pw_aff> auto &&fn) {
    auto val = visit(S->getOperand(0));

    if (val) {
      for (int i = 1; i < S->getNumOperands(); ++i) {
        auto inVal = visit(S->getOperand(i));
        if (inVal)
          val = fn(std::move(*val), std::move(*inVal));
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
    : public ConditionalVisitor<ConditionalAffinator, islpp::set> {

  ScevAffinator &affinator;
  ConditionalAffinator(ScevAffinator &aff) : affinator{aff} {}

  islpp::set visitAnd(llvm::Instruction &and_inst) override {
    auto lhs = llvm::dyn_cast<llvm::Instruction>(and_inst.getOperand(0));
    auto rhs = llvm::dyn_cast<llvm::Instruction>(and_inst.getOperand(1));
    assert(lhs);
    assert(rhs);
    return ISLPP_CHECK(visit(*lhs) * visit(*rhs));
  };

  islpp::set visitOr(llvm::Instruction &or_inst) override {
    auto lhs = llvm::dyn_cast<llvm::Instruction>(or_inst.getOperand(0));
    auto rhs = llvm::dyn_cast<llvm::Instruction>(or_inst.getOperand(1));
    assert(lhs);
    assert(rhs);
    return ISLPP_CHECK(visit(*lhs) + visit(*rhs));
  };

  islpp::set visitSelectInst(llvm::SelectInst &select) {
    auto selector = visitValue(select.getOperand(0));
    auto true_branch = visitValue(select.getOperand(1));
    auto false_branch = visitValue(select.getOperand(2));

    return selector * true_branch;
  }

  islpp::set visitICmpInst(llvm::ICmpInst &icmp) override {
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
        return ISLPP_CHECK(islpp::eq_set(*lhscev, *rhscev));
      case llvm::ICmpInst::Predicate::ICMP_NE:
        return ISLPP_CHECK(islpp::ne_set(*lhscev, *rhscev));
      case llvm::ICmpInst::Predicate::ICMP_UGE:
      case llvm::ICmpInst::Predicate::ICMP_SGE:
        return ISLPP_CHECK(islpp::ge_set(*lhscev, *rhscev));
      case llvm::ICmpInst::Predicate::ICMP_UGT:
      case llvm::ICmpInst::Predicate::ICMP_SGT:
        return ISLPP_CHECK(islpp::gt_set(*lhscev, *rhscev));
      case llvm::ICmpInst::Predicate::ICMP_ULE:
      case llvm::ICmpInst::Predicate::ICMP_SLE:
        return ISLPP_CHECK(islpp::le_set(*lhscev, *rhscev));
      case llvm::ICmpInst::Predicate::ICMP_ULT:
      case llvm::ICmpInst::Predicate::ICMP_SLT:
        return ISLPP_CHECK(islpp::lt_set(*lhscev, *rhscev));
        break;
      default:
        break;
      }
    }

    return ISLPP_CHECK(islpp::set{"{}"});
  }

  islpp::set visitInstruction(llvm::Instruction &) {
    return ISLPP_CHECK(islpp::set{"{}"});
  };

  islpp::set visitValue(llvm::Value *val) {
    if (auto instr = llvm::dyn_cast<llvm::Instruction>(val))
      return visit(instr);

    if (auto constant = llvm::dyn_cast<llvm::Constant>(val)) {
      assert(constant->getType()->getTypeID() == llvm::Type::IntegerTyID);
      auto val = constant->getUniqueInteger();

      if (val == 1)
        return ISLPP_CHECK(
            islpp::set{affinator.space.identity<islpp::multi_aff>()});
      else
        return ISLPP_CHECK(affinator.space.empty<islpp::set>());
    }

    return ISLPP_CHECK(islpp::set{"{}"});
  };
};

islpp::pw_aff valuate(golly::InstantiationVariable::Expr expr,
                      const AffineContext &context, llvm::ScalarEvolution &se,
                      islpp::space sp) {
  if (auto val = std::get_if<llvm::Value *>(&expr)) {
    auto affinator = ScevAffinator{
        .se = se,
        .context = context,
        .space = sp,
    };
    return *affinator.visit(se.getSCEV(*val));
  } else {
    return sp.constant<islpp::pw_aff>(std::get<int>(expr));
  }
}

islpp::multi_aff null_tuple(islpp::space space) {
  return islpp::multi_aff{isl_multi_aff_multi_val_on_space(
      space.yield(), isl_multi_val_read_from_str(islpp::ctx(), "{[]}"))};
}

islpp::set spacify(const AffineContext &context, llvm::ScalarEvolution &se) {
  islpp::set s{"{ [] }"};
  s = add_dims(std::move(s), islpp::dim::set, context.induction_vars.size());

  auto space = get_space(s);
  int i = 0;
  for (auto &[ptr, iv] : context.induction_vars) {
    auto lb = valuate(iv.lower_bound, context, se, space);
    auto ub = valuate(iv.upper_bound, context, se, space);
    auto identity = space.coeff<islpp::pw_aff>(islpp::dim::in, i++, 1);
    auto set = le_set(lb, identity) * lt_set(identity, ub);
    s = set * std::move(s);
  }

  return s;
}

islpp::set consolidate(llvm::Value *conditional, llvm::ScalarEvolution &se,
                       islpp::space space, const AffineContext &context) {
  ScevAffinator affinator{.se = se, .context = context, .space = space};
  // llvm::dbgs() << *conditional << "\n";
  if (auto instr = llvm::dyn_cast<llvm::Instruction>(conditional))
    return ISLPP_CHECK(ConditionalAffinator{affinator}.visit(*instr));
  else
    return islpp::set{"{}"};
}

struct PolyhedralBuilder {
  llvm::Function &f;
  ConditionalDominanceAnalysis &cda;
  SccOrdering &scc;
  StatementDetection &stmts;
  PscopDetection &detection;
  llvm::ScalarEvolution &se;
  llvm::LoopInfo &li;

  struct ThreadExpressions {
    islpp::map tau2cta;
    islpp::map tau2thread;
    islpp::map thread2warpTuple;
    islpp::map warpTuple2warp;
    islpp::map warpTuple2lane;
  };

  ThreadExpressions createThreadExpressions() {
    const auto &global = detection.getGlobalContext();

    // get space
    auto space = get_space(spacify(global, se));

    // create cta getter
    auto [tau2cta,
          tau2thread] = [&]() -> std::pair<islpp::multi_aff, islpp::multi_aff> {
      auto cta_expr = null_tuple(space);
      auto thd_expr = null_tuple(space);

      int index = 0;
      for (auto &[val, iv] : global.induction_vars) {
        if (iv.kind == InstantiationVariable::Kind::Block)
          cta_expr = flat_range_product(
              cta_expr,
              space.coeff<islpp::multi_aff>(islpp::dim::in, index, 1));

        if (iv.kind == InstantiationVariable::Kind::Thread)
          thd_expr = flat_range_product(
              thd_expr,
              space.coeff<islpp::multi_aff>(islpp::dim::in, index, 1));

        ++index;
      }

      return {cta_expr, thd_expr};
    }();

    auto warpSize = 32; // let some oracle determine this

    return ThreadExpressions{.tau2cta = islpp::map{tau2cta},
                             .tau2thread = islpp::map{tau2thread}};
  }

  islpp::union_map constructDomain() {
    llvm::DenseMap<const llvm::BasicBlock *, islpp::set> domains;

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
      if (auto cond = detection.getBranchInfo(br)) {
        auto true_set = ISLPP_CHECK(
            consolidate(br->getCondition(), se, space, loop_info->context));
        auto br_dims = islpp::dims(space, islpp::dim::set);
        for (auto &bb : cda.getTrueBranch(br)) {
          auto &dom = domains[bb];
          auto diff = islpp::dims(dom, islpp::dim::set) - br_dims;
          dom = add_dims(true_set, islpp::dim::set, diff) * dom;
        }

        for (auto &bb : cda.getFalseBranch(br)) {
          auto &dom = domains[bb];
          auto diff = islpp::dims(dom, islpp::dim::set) - br_dims;
          dom = dom - add_dims(true_set, islpp::dim::set, diff);
        }
      } else {
        // non-affine branch, introduce a param to distinguish taken and not
        // taken
        auto param = islpp::add_param(space, br->getNameOrAsOperand());
        auto param_count = islpp::dims(param, islpp::dim::param);
        // llvm::dbgs() << "param name: " << param << "\n";
        auto val =
            param.coeff<islpp::pw_aff>(islpp::dim::param, param_count - 1, 1);

        auto zero = param.constant<islpp::pw_aff>(0);
        auto one = param.constant<islpp::pw_aff>(1);

        auto true_set = eq_set(val, zero);
        auto false_set = eq_set(val, one);
        for (auto &bb : cda.getTrueBranch(br)) {
          auto &dom = domains[bb];
          dom = true_set * dom;
        }

        for (auto &bb : cda.getFalseBranch(br)) {
          auto &dom = domains[bb];
          dom = false_set * dom;
        }
      }
    }

    // apply the domain to all sets
    islpp::union_map ret{"{}"};

    for (auto &[bb, domain] : domains) {
      for (auto &stmt : stmts.iterateStatements(*bb)) {
        islpp::set s = name(islpp::set{"{[]}"}, stmt.getName());
        auto instances = name(domain, stmt.getName());
        ret = ret + islpp::union_map{unwrap(cross(s, instances))};
      }
    }

    return ret;
  }

  islpp::union_map constructDistribution(islpp::union_map domain) {
    const auto thread_dims = detection.getGlobalContext().induction_vars.size();
    // we need to project out the thread dims for all statements
    auto instances = range(domain);

    islpp::union_map ret;

    for_each(instances, [&ret, thread_dims](islpp::set set) {
      auto sp = get_space(set);
      std::vector<islpp::aff> affs;
      for (int i = 0; i < thread_dims; ++i)
        affs.emplace_back(sp.coeff<islpp::aff>(islpp::dim::in, i, 1));
      auto as_map = islpp::map{islpp::flat_range_product(affs)};
      ret = ret + islpp::union_map{domain_intersect(as_map, set)};
    });

    return ret;
  }

  islpp::union_map constructTemporalSchedule(islpp::union_map domain) {
    struct LoopTime {
      islpp::multi_aff prefix_expr;
      islpp::space space;
      int count = 0;
    };

    llvm::DenseMap<llvm::Loop *, LoopTime> times;
    auto space = get_space(spacify(detection.getGlobalContext(), se));

    // for time, we want to construct a null prefix time that we can build up
    times[nullptr] =
        LoopTime{.prefix_expr = null_tuple(space), .space = space, .count = 0};

    auto get_parent = [this](llvm::Loop *loop) -> llvm::Loop * {
      loop = loop->getParentLoop();
      while (loop) {
        if (detection.getLoopInfo(loop)->ivar_introduced)
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

    islpp::union_map ret{"{}"};
    scc.traverse([&](const llvm::BasicBlock *bb) {
      auto loop = li.getLoopFor(bb);
      auto loop_context = detection.getLoopInfo(loop);
      auto itr = times.find(loop_context->affine_loop);

      if (itr == times.end()) {
        // new affine loop
        auto &metadata = times[get_parent(loop)];
        auto my_count = metadata.count++;
        auto my_space = add_dims(metadata.space, islpp::dim::set, 1);
        auto my_expr = ([&]() {
          auto v =
              ISLPP_CHECK(add_dims(metadata.prefix_expr, islpp::dim::in, 1));
          auto c = ISLPP_CHECK(my_space.constant<islpp::multi_aff>(my_count));
          auto vc = ISLPP_CHECK(flat_range_product(std::move(v), std::move(c)));
          auto p = ISLPP_CHECK(my_space.coeff<islpp::multi_aff>(
              islpp::dim::in, dims(my_space, islpp::dim::set) - 1, 1));
          return ISLPP_CHECK(flat_range_product(std::move(vc), std::move(p)));
        })();

        times[loop_context->affine_loop] =
            LoopTime{.prefix_expr = my_expr, .space = my_space, .count = 0};
        itr = times.find(loop_context->affine_loop);
      }

      // loop already exists, increment the counter
      auto &metadata = itr->second;
      auto space = metadata.space;
      for (auto &elem : stmts.iterateStatements(*bb)) {
        auto my_count = metadata.count++;
        auto my_expr = ISLPP_CHECK(islpp::flat_range_product(
            metadata.prefix_expr, space.constant<islpp::multi_aff>(my_count)));

        while (dims(my_expr, islpp::dim::out) < max_depth)
          my_expr = flat_range_product(my_expr, space.zero<islpp::multi_aff>());

        auto my_map = islpp::map{name(my_expr, islpp::dim::in, elem.getName())};
        ret = ret + ISLPP_CHECK(islpp::union_map{my_map});
      }
    });

    ret = domain_intersect(ret, range(domain));

    return ret;
  }

  islpp::union_map constructValidBarriers(const ThreadExpressions &thread_exprs,
                                          islpp::union_map domain,
                                          islpp::union_map thread_alloc,
                                          islpp::union_map temporal_schedule) {
    // [S->T] -> Stmt
    islpp::union_map beta{"{}"};

    islpp::union_map rev_alloc = reverse(thread_alloc);
    scc.traverse([&](const llvm::BasicBlock *bb) {
      for (auto &stmt : stmts.iterateStatements(*bb)) {
        if (auto bar_stmt = stmt.as<golly::BarrierStatement>()) {
          islpp::union_set s{name(islpp::set{"{[]}"}, bar_stmt->getName())};
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

          const auto thread_pairs = universal(active_threads, active_threads) -
                                    identity(active_threads);

          if (auto warp_bar =
                  std::get_if<golly::BarrierStatement::Warp>(&barrier)) {
            warp_bar->mask;
            // do things with mask
          }

          if (auto block_bar =
                  std::get_if<golly::BarrierStatement::Block>(&barrier)) {
            // collect all blocks in this statement

            // inst -> tid
            auto dmn_insts =
                apply_range(domain_map(thread_pairs), tid_to_insts) +
                apply_range(range_map(thread_pairs), tid_to_insts);

            // filter to same gids
            auto same_gid =
                apply_range(islpp::union_map{thread_exprs.tau2cta},
                            reverse(islpp::union_map{thread_exprs.tau2cta}));

            beta = beta + domain_intersect(dmn_insts, wrap(same_gid));
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

  islpp::union_map constructSynchronizationSchedule(
      const ThreadExpressions &thd_exprs, islpp::union_map domain,
      islpp::union_map thread_allocation, islpp::union_map temporal_schedule) {
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

  std::pair<islpp::union_map, islpp::union_map>
  calculateAccessRelations(islpp::union_map domain) {
    islpp::union_map reads{"{}"};
    islpp::union_map writes{"{}"};

    scc.traverse([&](const llvm::BasicBlock *bb) {
      auto loop = li.getLoopFor(bb);
      auto &context = detection.getLoopInfo(loop)->context;
      auto space = get_space(spacify(context, se));
      auto affinator =
          ScevAffinator{.se = se, .context = context, .space = space};

      for (auto &stmt : stmts.iterateStatements(*bb)) {
        if (auto mem_acc = stmt.as<golly::MemoryAccessStatement>()) {
          auto ptr = mem_acc->getPointer();
          auto ptr_name = ptr->getName();

          if (auto offset = mem_acc->getOffset()) {
            if (auto val = affinator.visit(const_cast<llvm::Value *>(offset))) {
              auto map = name(name(islpp::map{*val}, islpp::dim::out, ptr_name),
                              islpp::dim::in, stmt.getName());
              if (mem_acc->getAccessType() ==
                  MemoryAccessStatement::Access::Read)
                reads = reads + islpp::union_map{map};
              else
                writes = writes + islpp::union_map{map};
            } else {
              // non-affine offset
              // ignore
            }
          } else {
            // there is no offset
            // treat it as 0
            auto zero = space.zero<islpp::pw_aff>();
            auto map = name(name(islpp::map{zero}, islpp::dim::out, ptr_name),
                            islpp::dim::in, stmt.getName());
            if (mem_acc->getAccessType() == MemoryAccessStatement::Access::Read)
              reads = reads + islpp::union_map{map};
            else {
              assert(mem_acc->getAccessType() ==
                     MemoryAccessStatement::Access::Write);
              writes = writes + islpp::union_map{map};
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