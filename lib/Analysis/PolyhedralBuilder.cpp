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
      return loop_expr * *step;
    }

    auto zero_start = se.getAddRecExpr(se.getConstant(start->getType(), 0),
                                       S->getStepRecurrence(se), S->getLoop(),
                                       S->getNoWrapFlags());

    auto res_expr = visit(zero_start);
    auto start_expr = visit(start);

    if (res_expr && start_expr)
      return *res_expr + *start_expr;
    else
      return llvm::None;
  }

  RetVal visitSequentialUMinExpr(const llvm::SCEVSequentialUMinExpr *S) {
    // todo
    return llvm::None;
  }
  RetVal visitUnknown(const llvm::SCEVUnknown *S) {
    const auto value = S->getValue();

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
    return visit(*lhs) * visit(*rhs);
  };

  islpp::set visitOr(llvm::Instruction &or_inst) override {
    auto lhs = llvm::dyn_cast<llvm::Instruction>(or_inst.getOperand(0));
    auto rhs = llvm::dyn_cast<llvm::Instruction>(or_inst.getOperand(1));
    assert(lhs);
    assert(rhs);
    return visit(*lhs) + visit(*rhs);
  };

  islpp::set visitICmpInst(llvm::ICmpInst &icmp) override {
    llvm::dbgs() << icmp << "\n";
    auto lhs = icmp.getOperand(0);
    auto rhs = icmp.getOperand(1);
    assert(lhs);
    assert(rhs);

    auto lhscev = affinator.visit(lhs);
    auto rhscev = affinator.visit(rhs);

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

    return ISLPP_CHECK(islpp::set{"{}"});
  }

  islpp::set visitInstruction(llvm::Instruction &) { return islpp::set{"{}"}; };
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

  islpp::union_map constructDomain() {
    llvm::DenseMap<const llvm::BasicBlock *, islpp::set> domains;

    scc.traverse([&](const llvm::BasicBlock *bb) {
      auto loop = li.getLoopFor(bb);
      auto loop_info = detection.getLoopInfo(loop);
      auto domain = spacify(loop_info->context, se);

      llvm::dbgs() << bb->getName() << domain << "\n";
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
        llvm::dbgs() << "param name: " << param << "\n";
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
      islpp::set domain{"{ [] }"};
      int time = 0;
    };

    llvm::DenseMap<llvm::Loop *, LoopTime> times;
    times[nullptr] = LoopTime{};

    scc.traverse([&](const llvm::BasicBlock *bb) {
      auto loop = li.getLoopFor(bb);
      if (auto itr = times.find(loop); itr != times.end()) {
        // new loop, introduce new time
      } else {
        auto parent = loop->getParentLoop();
      }
    });

    return islpp::union_map{"{}"};
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
  const auto domain = builder.constructDomain();
  const auto thread_alloc = builder.constructDistribution(domain);
  const auto temporal_shedule = builder.constructTemporalSchedule(domain);
  const auto [reads, writes] = builder.calculateAccessRelations(domain);

  return Pscop{
      .instantiation_domain = domain,
      .thread_allocation = thread_alloc,
      .temporal_schedule = temporal_shedule,
      .sync_schedule = islpp::union_map{"{}"},
      .write_access_relation = writes,
      .read_access_relation = reads,
  };
}
} // namespace golly