#include <algorithm>
#include <golly/Analysis/DimensionDetection.h>
#include <golly/Analysis/PscopDetection.h>
#include <iostream>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/RegionInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/IR/Function.h>

namespace golly {
using llvm::ArrayRef;
using llvm::LoopAnalysis;
using llvm::LoopInfo;
using llvm::Region;
using llvm::RegionInfo;
using llvm::RegionInfoAnalysis;
using llvm::ScalarEvolution;
using llvm::ScalarEvolutionAnalysis;
using llvm::SCEV;
using llvm::SetVector;
using llvm::SmallDenseSet;

// scev accessors heavily modified from llvm-project/polly/lib/SCEVValidator.cpp
// for the most part, we do not use loops to find our induction variables
enum class ScevType {
  Int,
  Param,
  InductionVar,
  Invalid,
};

class ScevAffineExpression {
  ScevType type;
  SetVector<const SCEV *> params;

public:
  explicit ScevAffineExpression(ScevType type, const SCEV *expr = nullptr)
      : type{type} {
    if (expr)
      params.insert(expr);
  }

  bool isConstant() const { return type <= ScevType::Param; }
  ScevType getType() const { return type; }
  ArrayRef<const SCEV *> getParams() const { return params.getArrayRef(); }

  void addParams(const SCEV *expr) { params.insert(expr); }
  void addParams(const ScevAffineExpression &rhs) {
    params.insert(rhs.params.begin(), rhs.params.end());
  }

  explicit operator bool() const { return type != ScevType::Invalid; }

  void merge(const ScevAffineExpression &rhs) {
    type = std::max(type, rhs.type);
    addParams(rhs);
  }
};

struct ScevValidator : llvm::SCEVVisitor<ScevValidator, ScevAffineExpression> {
  ScalarEvolution &scalar_evolution;
  llvm::Region *region;
  llvm::Loop *scope;

  ScevValidator(ScalarEvolution &se, llvm::Region *reg, llvm::Loop *scope)
      : scalar_evolution{se}, region{reg}, scope{scope} {}

  ScevAffineExpression visitConstant(const llvm::SCEVConstant *) {
    return ScevAffineExpression(ScevType::Int);
  }

  ScevAffineExpression visitZeroExtendOrTruncateExpr(const SCEV *expr,
                                                     const SCEV *operand) {
    auto op = visit(operand);
    auto type = op.getType();

    if (type == ScevType::Invalid)
      return ScevAffineExpression(ScevType::Invalid);
    return ScevAffineExpression(ScevType::Param, expr);
  }

  ScevAffineExpression visitPtrToIntExpr(const llvm::SCEVPtrToIntExpr *ptr) {
    return visit(ptr->getOperand());
  }

  ScevAffineExpression visitTruncateExpr(const llvm::SCEVTruncateExpr *expr) {
    return visitZeroExtendOrTruncateExpr(expr, expr->getOperand());
  }

  ScevAffineExpression
  visitZeroExtendExpr(const llvm::SCEVZeroExtendExpr *expr) {
    return visitZeroExtendOrTruncateExpr(expr, expr->getOperand());
  }

  ScevAffineExpression
  visitSignExtendExpr(const llvm::SCEVSignExtendExpr *expr) {
    return visit(expr->getOperand());
  }

  ScevAffineExpression visitAddExpr(const llvm::SCEVAddExpr *expr) {
    return mergeOperands(expr);
  }

  ScevAffineExpression visitMulExpr(const llvm::SCEVMulExpr *expr) {
    ScevAffineExpression res(ScevType::Int);

    bool multiParam = false;

    for (auto &elem : expr->operands()) {
      auto op = visit(elem);

      switch (op.getType()) {
      case ScevType::Int:
        continue;
      case ScevType::Param:
        if (res.getType() == ScevType::Param) {
          multiParam = true;
          continue;
        }
        // fallthrough
      case ScevType::InductionVar: {
        // more than one non-int operand in mul expr
        if (res.getType() != ScevType::Int) {
          return ScevAffineExpression(ScevType::Invalid);
        }
        res.merge(op);
      }
      }

      if (!res) {
        break;
      }
    }

    if (multiParam && res) {
      return ScevAffineExpression(ScevType::Param, expr);
    }

    return res;
  }

  ScevAffineExpression visitAddRecExpr(const llvm::SCEVAddRecExpr *expr) {
    llvm::dbgs() << "add rec\n";
    if (!expr->isAffine())
      return ScevAffineExpression(ScevType::Invalid);

    auto start = visit(expr->getStart());
    auto recc = visit(expr->getStepRecurrence(scalar_evolution));

    if (!start)
      return start;

    if (!recc)
      return recc;

    // THE ACTUAL MEAT PART I NEED TO GET

    const auto loopOfExpr = expr->getLoop();
    if (region->contains(loopOfExpr)) {
      if (!scope || !loopOfExpr->contains(scope)) {
        // non-affine subregion
        return ScevAffineExpression(ScevType::Invalid);
      }

      if (recc.getType() == ScevType::Int) {
        auto res = ScevAffineExpression(ScevType::InductionVar);
        res.addParams(start);
        return res;
      }
      return ScevAffineExpression(ScevType::Invalid);
    }

    llvm::dbgs() << "failed\n";
    return ScevAffineExpression(ScevType::Invalid);
  }

  ScevAffineExpression visitSMaxExpr(const llvm::SCEVSMaxExpr *expr) {
    return mergeOperands(expr);
  }

  ScevAffineExpression visitSMinExpr(const llvm::SCEVSMinExpr *expr) {
    return mergeOperands(expr);
  }

  ScevAffineExpression visitUMaxExpr(const llvm::SCEVUMaxExpr *expr) {
    return allOperands(expr, [](const ScevAffineExpression &expr) {
      return expr.isConstant();
    });
  }

  ScevAffineExpression visitUMinExpr(const llvm::SCEVUMinExpr *expr) {
    return allOperands(expr, [](const ScevAffineExpression &expr) {
      return expr.isConstant();
    });
  }

  ScevAffineExpression
  visitSequentialUMinExpr(const llvm::SCEVSequentialUMinExpr *expr) {
    return allOperands(expr, [](const ScevAffineExpression &expr) {
      return expr.isConstant();
    });
  }

  ScevAffineExpression visitGenericInst(llvm::Instruction *i,
                                        const llvm::SCEV *s) {
    // hopefully the instruction is outside the region
    // if (R->contains(I)) {
    //   LLVM_DEBUG(dbgs() << "INVALID: UnknownExpr references an instruction "
    //                        "within the region\n");
    //   return ValidatorResult(SCEVType::INVALID);
    // }

    return ScevAffineExpression(ScevType::Param, s);
  }

  ScevAffineExpression visitLoadInstruction(llvm::Instruction *i,
                                            const llvm::SCEV *s) {
    // if (R->contains(I) && ILS) {
    //   ILS->insert(cast<LoadInst>(I));
    //   return ValidatorResult(SCEVType::PARAM, S);
    // }

    return visitGenericInst(i, s);
  }

  ScevAffineExpression visitDivision(const llvm::SCEV *dividend,
                                     const llvm::SCEV *divisor,
                                     const llvm::SCEV *divExpr,
                                     llvm::Instruction *sDiv = nullptr) {

    // a constant is divisible
    if (llvm::isa<llvm::SCEVConstant>(divisor) && !divisor->isZero()) {
      return visit(dividend);
    }

    if (sDiv) {
      return visitGenericInst(sDiv, divExpr);
    }

    auto lhs = visit(dividend);
    auto rhs = visit(divisor);

    if (lhs.isConstant() && rhs.isConstant())
      return ScevAffineExpression(ScevType::Param, divExpr);

    // unsigned div of non-const exprs
    return ScevAffineExpression(ScevType::Invalid);
  }

  ScevAffineExpression visitUDivExpr(const llvm::SCEVUDivExpr *expr) {
    auto dividend = expr->getLHS();
    auto divisor = expr->getRHS();
    return visitDivision(dividend, divisor, expr);
  }

  ScevAffineExpression visitSDivInstruction(llvm::Instruction *sDiv,
                                            const SCEV *expr) {
    auto dividend = scalar_evolution.getSCEV(sDiv->getOperand(0));
    auto divisor = scalar_evolution.getSCEV(sDiv->getOperand(1));
    return visitDivision(dividend, divisor, expr, sDiv);
  }

  ScevAffineExpression visitSRemInstruction(llvm::Instruction *sRem,
                                            const SCEV *expr) {
    auto divisor = sRem->getOperand(1);

    auto *ci = dyn_cast<llvm::ConstantInt>(divisor);
    if (!ci || ci->isZeroValue())
      return visitGenericInst(sRem, expr);

    auto dividend = scalar_evolution.getSCEV(sRem->getOperand(0));

    return visit(dividend);
  }

  ScevAffineExpression visitCall(llvm::Instruction *instr, const SCEV *expr) {
    auto *call = llvm::dyn_cast<llvm::CallInst>(instr);
    if (call) {
      const auto callee = call->getCalledFunction()->getGlobalIdentifier();
      llvm::dbgs() << "callee: " << callee << "\n";
      if (callee == "llvm.nvvm.read.ptx.sreg.tid.x") {
        return ScevAffineExpression(ScevType::InductionVar, expr);
      }
    }
    return visitGenericInst(instr, expr);
  }

  ScevAffineExpression visitUnknown(const llvm::SCEVUnknown *Expr) {
    llvm::dbgs() << "unknown\n";
    auto *V = Expr->getValue();

    if (!Expr->getType()->isIntegerTy() && !Expr->getType()->isPointerTy()) {
      return ScevAffineExpression(ScevType::Invalid);
    }

    if (llvm::isa<llvm::UndefValue>(V)) {
      return ScevAffineExpression(ScevType::Invalid);
    }

    if (llvm::Instruction *I = dyn_cast<llvm::Instruction>(Expr->getValue())) {
      switch (I->getOpcode()) {
      // case llvm::Instruction::IntToPtr:
      // return visit(SE.getSCEVAtScope(I->getOperand(0), Scope));
      case llvm::Instruction::Call:
        return visitCall(I, Expr); // looking for intrinsics
      case llvm::Instruction::Load:
        return visitLoadInstruction(I, Expr);
      case llvm::Instruction::SDiv:
        return visitSDivInstruction(I, Expr);
      case llvm::Instruction::SRem:
        return visitSRemInstruction(I, Expr);
      default:
        return visitGenericInst(I, Expr);
      }
    }

    if (Expr->getType()->isPointerTy()) {
      if (isa<llvm::ConstantPointerNull>(V))
        return ScevAffineExpression(ScevType::Int); // "int"
    }

    return ScevAffineExpression(ScevType::Param, Expr);
  }

  template <typename T> ScevAffineExpression mergeOperands(const T *expr) {
    ScevAffineExpression res(ScevType::Int);

    for (auto &elem : expr->operands()) {
      auto op = visit(elem);
      res.merge(op);

      if (!res) {
        break;
      }
    }

    return res;
  }

  template <typename T, typename F>
  ScevAffineExpression allOperands(const T *expr, F &&func) {
    for (auto &op : expr->operands()) {
      if (!func(visit(op))) {
        return ScevAffineExpression(ScevType::Invalid);
      }
    }
    return ScevAffineExpression(ScevType::Param, expr);
  }
};

AnalysisKey PscopDetectionPass::Key;

struct PscopDetection {
  DetectedIntrinsics &dimensions;
  LoopInfo &loop_info;
  RegionInfo &region_info;
  ScalarEvolution &scalar_evolution;

  SmallDenseSet<llvm::Value *> inductionVars;

  PscopDetection(DetectedIntrinsics &dims, ScalarEvolution &scev, LoopInfo &l,
                 RegionInfo &r)
      : dimensions{dims}, loop_info{l}, region_info{r}, scalar_evolution{scev} {
  }

  void run() {
    for (auto &&[instr, type] : dimensions.detections) {
      if (std::ranges::find(threadIds, type) != threadIds.end()) {
        inductionVars.insert(instr);
      }
    }

    scalar_evolution.print(llvm::dbgs());
    const auto r = region_info.getTopLevelRegion();
    for (auto &&bb : r->blocks()) {
      for (auto &&instr : *bb) {
        llvm::dbgs() << instr << "\n";
        if (const auto ptr =
                llvm::dyn_cast_or_null<llvm::GetElementPtrInst>(&instr)) {

          const auto scev = scalar_evolution.getSCEV(ptr->getOperand(1));

          // polly::SCEVValidator validator{r, nullptr, scalar_evolution,
          // nullptr}; auto res = validator.visit(scev);

          ScevValidator validator{scalar_evolution, r, nullptr};
          auto res = validator.visit(scev);
          llvm::dbgs() << instr << " is " << (!res ? "not" : "")
                       << " affine!\n";
          for (auto &elem : res.getParams()) {
            llvm::dbgs() << "param:" << *elem << "\n";
          }
        }
      }
    }

    findPscops(*region_info.getTopLevelRegion());
  }

  void findPscops(llvm::Region &r) {
    // is this region a branch?

    // is this region a loop?
    if (auto loop = loop_info.getLoopFor(r.getEntry())) {
      llvm::dbgs() << r << " is a loop\n";
      int i = 0;
      for (auto &&bb : r.blocks()) {
        llvm::dbgs() << bb->getName() << " " << i++ << "\n";
      }

      // extract pw_aff bounds
      loop->dump();
      llvm::dbgs() << '\n';

      const auto cmp = loop->getLatchCmpInst();
      const auto loopVar = loop->getCanonicalInductionVariable();

      inductionVars.insert(loopVar);

      if (const auto bounds = loop->getBounds(scalar_evolution)) {
        llvm::dbgs() << bounds->getInitialIVValue() << " "
                     << bounds->getFinalIVValue() << " "
                     << *bounds->getStepValue() << "\n";
        if (isPiecewiseAffine(bounds->getInitialIVValue()) &&
            isPiecewiseAffine(bounds->getFinalIVValue())) {
          llvm::dbgs() << r << " has valid loop bounds";
        }
      }
    }

    for (const auto *bb : r.blocks()) {
    }

    for (auto &subregion : r) {
      findPscops(*subregion);
    }
  }

  bool isPiecewiseAffine(llvm::Value &val) {
    const auto scev = scalar_evolution.getSCEV(&val);
    if (!scev)
      return false;

    // ScevValidator validator;
    // validator.visit(scev);

    return false;
  };
};

PscopDetectionPass::Result
PscopDetectionPass::run(Function &f, FunctionAnalysisManager &am) {
  auto &dd = am.getResult<DimensionDetection>(f);
  auto &se = am.getResult<ScalarEvolutionAnalysis>(f);
  auto &la = am.getResult<LoopAnalysis>(f);
  auto &ri = am.getResult<RegionInfoAnalysis>(f);

  PscopDetection pscops{dd, se, la, ri};
  pscops.run();

  return {};
}

PreservedAnalyses RunPscopDetection::run(Function &f,
                                         FunctionAnalysisManager &am) {
  am.getResult<PscopDetectionPass>(f);
  return PreservedAnalyses::all();
}

} // namespace golly
