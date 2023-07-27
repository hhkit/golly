//===--- SCEVValidator.h - Detect Scops -------------------------*- C++ -*-===//
//
// Modified from a similar file from the LLVM Projecet.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Checks if a SCEV expression represents a valid affine expression.
//===----------------------------------------------------------------------===//

#ifndef GOLLY_SUPPORT_SCEVVALIDATOR_H
#define GOLLY_SUPPORT_SCEVVALIDATOR_H

#include <golly/Support/ScopHelper.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/RegionInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/Support/Debug.h>

namespace polly {
using namespace llvm;

#define DEBUG_TYPE "polly-scev-validator"

namespace SCEVType {
/// The type of a SCEV
///
/// To check for the validity of a SCEV we assign to each SCEV a type. The
/// possible types are INT, PARAM, IV and INVALID. The order of the types is
/// important. The subexpressions of SCEV with a type X can only have a type
/// that is smaller or equal than X.
enum TYPE {
  // An integer value.
  INT,

  // An expression that is constant during the execution of the Scop,
  // but that may depend on parameters unknown at compile time.
  PARAM,

  // An expression that may change during the execution of the SCoP.
  IV,

  // An invalid expression.
  INVALID
};
} // namespace SCEVType

/// The result the validator returns for a SCEV expression.
class ValidatorResult final {
  /// The type of the expression
  SCEVType::TYPE Type;

  /// The set of Parameters in the expression.
  ParameterSetTy Parameters;

public:
  /// The copy constructor
  ValidatorResult(const ValidatorResult &Source) {
    Type = Source.Type;
    Parameters = Source.Parameters;
  }

  /// Construct a result with a certain type and no parameters.
  ValidatorResult(SCEVType::TYPE Type) : Type(Type) {
    assert(Type != SCEVType::PARAM && "Did you forget to pass the parameter");
  }

  /// Construct a result with a certain type and a single parameter.
  ValidatorResult(SCEVType::TYPE Type, const SCEV *Expr) : Type(Type) {
    Parameters.insert(Expr);
  }

  /// Get the type of the ValidatorResult.
  SCEVType::TYPE getType() { return Type; }

  /// Is the analyzed SCEV constant during the execution of the SCoP.
  bool isConstant() { return Type == SCEVType::INT || Type == SCEVType::PARAM; }

  /// Is the analyzed SCEV valid.
  bool isValid() { return Type != SCEVType::INVALID; }

  /// Is the analyzed SCEV of Type IV.
  bool isIV() { return Type == SCEVType::IV; }

  /// Is the analyzed SCEV of Type INT.
  bool isINT() { return Type == SCEVType::INT; }

  /// Is the analyzed SCEV of Type PARAM.
  bool isPARAM() { return Type == SCEVType::PARAM; }

  /// Get the parameters of this validator result.
  const ParameterSetTy &getParameters() { return Parameters; }

  /// Add the parameters of Source to this result.
  void addParamsFrom(const ValidatorResult &Source) {
    Parameters.insert(Source.Parameters.begin(), Source.Parameters.end());
  }

  /// Merge a result.
  ///
  /// This means to merge the parameters and to set the Type to the most
  /// specific Type that matches both.
  void merge(const ValidatorResult &ToMerge) {
    Type = std::max(Type, ToMerge.Type);
    addParamsFrom(ToMerge);
  }

  void print(raw_ostream &OS) {
    switch (Type) {
    case SCEVType::INT:
      OS << "SCEVType::INT";
      break;
    case SCEVType::PARAM:
      OS << "SCEVType::PARAM";
      break;
    case SCEVType::IV:
      OS << "SCEVType::IV";
      break;
    case SCEVType::INVALID:
      OS << "SCEVType::INVALID";
      break;
    }
  }
};

static raw_ostream &operator<<(raw_ostream &OS, ValidatorResult &VR) {
  VR.print(OS);
  return OS;
}

/// Check if a SCEV is valid in a SCoP.
class SCEVValidator : public SCEVVisitor<SCEVValidator, ValidatorResult> {
private:
  const Region *R;
  Loop *Scope;
  ScalarEvolution &SE;
  InvariantLoadsSetTy *ILS;

public:
  SCEVValidator(const Region *R, Loop *Scope, ScalarEvolution &SE,
                InvariantLoadsSetTy *ILS)
      : R(R), Scope(Scope), SE(SE), ILS(ILS) {}

  ValidatorResult visitConstant(const SCEVConstant *Constant) {
    return ValidatorResult(SCEVType::INT);
  }

  ValidatorResult visitZeroExtendOrTruncateExpr(const SCEV *Expr,
                                                const SCEV *Operand) {
    ValidatorResult Op = visit(Operand);
    auto Type = Op.getType();

    // If unsigned operations are allowed return the operand, otherwise
    // check if we can model the expression without unsigned assumptions.
    // if (PollyAllowUnsignedOperations || Type == SCEVType::INVALID)
    //   return Op;

    if (Type == SCEVType::IV)
      return ValidatorResult(SCEVType::INVALID);
    return ValidatorResult(SCEVType::PARAM, Expr);
  }

  ValidatorResult visitPtrToIntExpr(const SCEVPtrToIntExpr *Expr) {
    return visit(Expr->getOperand());
  }

  ValidatorResult visitTruncateExpr(const SCEVTruncateExpr *Expr) {
    return visitZeroExtendOrTruncateExpr(Expr, Expr->getOperand());
  }

  ValidatorResult visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
    return visitZeroExtendOrTruncateExpr(Expr, Expr->getOperand());
  }

  ValidatorResult visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
    return visit(Expr->getOperand());
  }

  ValidatorResult visitAddExpr(const SCEVAddExpr *Expr) {
    ValidatorResult Return(SCEVType::INT);

    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));
      Return.merge(Op);

      // Early exit.
      if (!Return.isValid())
        break;
    }

    return Return;
  }

  ValidatorResult visitMulExpr(const SCEVMulExpr *Expr) {
    ValidatorResult Return(SCEVType::INT);

    bool HasMultipleParams = false;

    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));

      if (Op.isINT())
        continue;

      if (Op.isPARAM() && Return.isPARAM()) {
        HasMultipleParams = true;
        continue;
      }

      if ((Op.isIV() || Op.isPARAM()) && !Return.isINT()) {
        LLVM_DEBUG(
            dbgs() << "INVALID: More than one non-int operand in MulExpr\n"
                   << "\tExpr: " << *Expr << "\n"
                   << "\tPrevious expression type: " << Return << "\n"
                   << "\tNext operand (" << Op << "): " << *Expr->getOperand(i)
                   << "\n");

        return ValidatorResult(SCEVType::INVALID);
      }

      Return.merge(Op);
    }

    if (HasMultipleParams && Return.isValid())
      return ValidatorResult(SCEVType::PARAM, Expr);

    return Return;
  }

  ValidatorResult visitAddRecExpr(const SCEVAddRecExpr *Expr) {
    if (!Expr->isAffine()) {
      LLVM_DEBUG(dbgs() << "INVALID: AddRec is not affine");
      return ValidatorResult(SCEVType::INVALID);
    }

    ValidatorResult Start = visit(Expr->getStart());
    ValidatorResult Recurrence = visit(Expr->getStepRecurrence(SE));

    if (!Start.isValid())
      return Start;

    if (!Recurrence.isValid())
      return Recurrence;

    auto *L = Expr->getLoop();
    if (R->contains(L) && (!Scope || !L->contains(Scope))) {
      LLVM_DEBUG(
          dbgs() << "INVALID: Loop of AddRec expression boxed in an a "
                    "non-affine subregion or has a non-synthesizable exit "
                    "value.");
      return ValidatorResult(SCEVType::INVALID);
    }

    if (R->contains(L)) {
      if (Recurrence.isINT()) {
        ValidatorResult Result(SCEVType::IV);
        Result.addParamsFrom(Start);
        return Result;
      }

      LLVM_DEBUG(dbgs() << "INVALID: AddRec within scop has non-int"
                           "recurrence part");
      return ValidatorResult(SCEVType::INVALID);
    }

    assert(Recurrence.isConstant() && "Expected 'Recurrence' to be constant");

    // Directly generate ValidatorResult for Expr if 'start' is zero.
    if (Expr->getStart()->isZero())
      return ValidatorResult(SCEVType::PARAM, Expr);

    // Translate AddRecExpr from '{start, +, inc}' into 'start + {0, +, inc}'
    // if 'start' is not zero.
    const SCEV *ZeroStartExpr = SE.getAddRecExpr(
        SE.getConstant(Expr->getStart()->getType(), 0),
        Expr->getStepRecurrence(SE), Expr->getLoop(), Expr->getNoWrapFlags());

    ValidatorResult ZeroStartResult =
        ValidatorResult(SCEVType::PARAM, ZeroStartExpr);
    ZeroStartResult.addParamsFrom(Start);

    return ZeroStartResult;
  }

  ValidatorResult visitSMaxExpr(const SCEVSMaxExpr *Expr) {
    ValidatorResult Return(SCEVType::INT);

    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));

      if (!Op.isValid())
        return Op;

      Return.merge(Op);
    }

    return Return;
  }

  ValidatorResult visitSMinExpr(const SCEVSMinExpr *Expr) {
    ValidatorResult Return(SCEVType::INT);

    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));

      if (!Op.isValid())
        return Op;

      Return.merge(Op);
    }

    return Return;
  }

  ValidatorResult visitUMaxExpr(const SCEVUMaxExpr *Expr) {
    // We do not support unsigned max operations. If 'Expr' is constant during
    // Scop execution we treat this as a parameter, otherwise we bail out.
    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));

      if (!Op.isConstant()) {
        LLVM_DEBUG(dbgs() << "INVALID: UMaxExpr has a non-constant operand");
        return ValidatorResult(SCEVType::INVALID);
      }
    }

    return ValidatorResult(SCEVType::PARAM, Expr);
  }

  ValidatorResult visitUMinExpr(const SCEVUMinExpr *Expr) {
    // We do not support unsigned min operations. If 'Expr' is constant during
    // Scop execution we treat this as a parameter, otherwise we bail out.
    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));

      if (!Op.isConstant()) {
        LLVM_DEBUG(dbgs() << "INVALID: UMinExpr has a non-constant operand");
        return ValidatorResult(SCEVType::INVALID);
      }
    }

    return ValidatorResult(SCEVType::PARAM, Expr);
  }

  ValidatorResult visitSequentialUMinExpr(const SCEVSequentialUMinExpr *Expr) {
    // We do not support unsigned min operations. If 'Expr' is constant during
    // Scop execution we treat this as a parameter, otherwise we bail out.
    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));

      if (!Op.isConstant()) {
        LLVM_DEBUG(
            dbgs()
            << "INVALID: SCEVSequentialUMinExpr has a non-constant operand");
        return ValidatorResult(SCEVType::INVALID);
      }
    }

    return ValidatorResult(SCEVType::PARAM, Expr);
  }

  ValidatorResult visitGenericInst(Instruction *I, const SCEV *S) {
    if (R->contains(I)) {
      LLVM_DEBUG(dbgs() << "INVALID: UnknownExpr references an instruction "
                           "within the region\n");
      return ValidatorResult(SCEVType::INVALID);
    }

    return ValidatorResult(SCEVType::PARAM, S);
  }

  ValidatorResult visitLoadInstruction(Instruction *I, const SCEV *S) {
    if (R->contains(I) && ILS) {
      ILS->insert(cast<LoadInst>(I));
      return ValidatorResult(SCEVType::PARAM, S);
    }

    return visitGenericInst(I, S);
  }

  ValidatorResult visitDivision(const SCEV *Dividend, const SCEV *Divisor,
                                const SCEV *DivExpr,
                                Instruction *SDiv = nullptr) {

    // First check if we might be able to model the division, thus if the
    // divisor is constant. If so, check the dividend, otherwise check if
    // the whole division can be seen as a parameter.
    if (isa<SCEVConstant>(Divisor) && !Divisor->isZero())
      return visit(Dividend);

    // For signed divisions use the SDiv instruction to check for a parameter
    // division, for unsigned divisions check the operands.
    if (SDiv)
      return visitGenericInst(SDiv, DivExpr);

    ValidatorResult LHS = visit(Dividend);
    ValidatorResult RHS = visit(Divisor);
    if (LHS.isConstant() && RHS.isConstant())
      return ValidatorResult(SCEVType::PARAM, DivExpr);

    LLVM_DEBUG(
        dbgs() << "INVALID: unsigned division of non-constant expressions");
    return ValidatorResult(SCEVType::INVALID);
  }

  ValidatorResult visitUDivExpr(const SCEVUDivExpr *Expr) {
    // if (!PollyAllowUnsignedOperations)
    //   return ValidatorResult(SCEVType::INVALID);

    auto *Dividend = Expr->getLHS();
    auto *Divisor = Expr->getRHS();
    return visitDivision(Dividend, Divisor, Expr);
  }

  ValidatorResult visitSDivInstruction(Instruction *SDiv, const SCEV *Expr) {
    assert(SDiv->getOpcode() == Instruction::SDiv &&
           "Assumed SDiv instruction!");

    auto *Dividend = SE.getSCEV(SDiv->getOperand(0));
    auto *Divisor = SE.getSCEV(SDiv->getOperand(1));
    return visitDivision(Dividend, Divisor, Expr, SDiv);
  }

  ValidatorResult visitSRemInstruction(Instruction *SRem, const SCEV *S) {
    assert(SRem->getOpcode() == Instruction::SRem &&
           "Assumed SRem instruction!");

    auto *Divisor = SRem->getOperand(1);
    auto *CI = dyn_cast<ConstantInt>(Divisor);
    if (!CI || CI->isZeroValue())
      return visitGenericInst(SRem, S);

    auto *Dividend = SRem->getOperand(0);
    auto *DividendSCEV = SE.getSCEV(Dividend);
    return visit(DividendSCEV);
  }

  ValidatorResult visitUnknown(const SCEVUnknown *Expr) {
    Value *V = Expr->getValue();

    if (!Expr->getType()->isIntegerTy() && !Expr->getType()->isPointerTy()) {
      LLVM_DEBUG(dbgs() << "INVALID: UnknownExpr is not an integer or pointer");
      return ValidatorResult(SCEVType::INVALID);
    }

    if (isa<UndefValue>(V)) {
      LLVM_DEBUG(dbgs() << "INVALID: UnknownExpr references an undef value");
      return ValidatorResult(SCEVType::INVALID);
    }

    if (Instruction *I = dyn_cast<Instruction>(Expr->getValue())) {
      switch (I->getOpcode()) {
      case Instruction::IntToPtr:
        return visit(SE.getSCEVAtScope(I->getOperand(0), Scope));
      case Instruction::Load:
        return visitLoadInstruction(I, Expr);
      case Instruction::SDiv:
        return visitSDivInstruction(I, Expr);
      case Instruction::SRem:
        return visitSRemInstruction(I, Expr);
      default:
        return visitGenericInst(I, Expr);
      }
    }

    if (Expr->getType()->isPointerTy()) {
      if (isa<ConstantPointerNull>(V))
        return ValidatorResult(SCEVType::INT); // "int"
    }

    return ValidatorResult(SCEVType::PARAM, Expr);
  }
};
} // namespace polly
#endif /* GOLLY_SUPPORT_SCEVVALIDATOR_H */
