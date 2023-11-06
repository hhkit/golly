#ifndef ANALYSIS_POLYHEDRALBUILDERSUPPORT_H
#define ANALYSIS_POLYHEDRALBUILDERSUPPORT_H

#include "golly/Analysis/PscopDetector.h"
#include "golly/Support/ConditionalVisitor.h"
#include "golly/Support/isl_llvm.h"
#include <llvm/ADT/Optional.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>

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

} // namespace golly

#endif /* ANALYSIS_POLYHEDRALBUILDERSUPPORT_H */
