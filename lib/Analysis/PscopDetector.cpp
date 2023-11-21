#include "golly/Analysis/PscopDetector.h"
#include "golly/Analysis/ConditionalDominanceAnalysis.h"
#include "golly/Analysis/CudaParameterDetection.h"
#include "golly/Analysis/SccOrdering.h"
#include "golly/Support/ConditionalAtomizer.h"
#include "golly/Support/ConditionalVisitor.h"

#include <llvm/ADT/SetOperations.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/RegionInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>

namespace golly {

enum class ExprClass {
  Constant,
  IVar,
  Param,
  Invalid,
};

struct ScevValidator : llvm::SCEVVisitor<ScevValidator, ExprClass> {
  using RetVal = ExprClass;

  llvm::ScalarEvolution &se;

  const AffineContext &context;

  RetVal visitConstant(const llvm::SCEVConstant *cint) {
    return ExprClass::Constant;
  }

  RetVal visitAddExpr(const llvm::SCEVAddExpr *S) { return mergeNary(S); }

  RetVal visitMulExpr(const llvm::SCEVMulExpr *S) {
    auto lhs = visit(S->getOperand(0));
    auto rhs = visit(S->getOperand(1));

    // check for invalid
    auto newClass = std::max(lhs, rhs);

    if (newClass == ExprClass::Invalid)
      return ExprClass::Invalid;

    if (lhs == ExprClass::Param) {
      if (rhs == ExprClass::IVar)
        return ExprClass::Invalid;

      // else expr is param * cint or param * param which is param
      return ExprClass::Param;
    }

    if (lhs == ExprClass::IVar) {
      if (rhs == ExprClass::Constant)
        return ExprClass::IVar;

      // else expr is ivar * param or ivar * ivar which is invalid
      return ExprClass::Invalid;
    }

    // lhs must be CInt
    return rhs;
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
  RetVal visitSMaxExpr(const llvm::SCEVSMaxExpr *S) { return mergeNary(S); }
  RetVal visitUMaxExpr(const llvm::SCEVUMaxExpr *S) { return mergeNary(S); }
  RetVal visitSMinExpr(const llvm::SCEVSMinExpr *S) { return mergeNary(S); }
  RetVal visitUMinExpr(const llvm::SCEVUMinExpr *S) { return mergeNary(S); }

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
    return ExprClass::Invalid;
  }

  RetVal visitAddRecExpr(const llvm::SCEVAddRecExpr *S) {
    if (!S->isAffine())
      return ExprClass::Invalid;

    const auto start = S->getStart();
    const auto step = S->getStepRecurrence(se);

    if (start->isZero()) {
      // todo
      auto step = visit(S->getOperand(1));
      // loop MUST exist
      auto indvar = S->getLoop()->getCanonicalInductionVariable();

      if (!indvar) // no canonical indvar
        return ExprClass::Invalid;

      if (step != ExprClass::Constant)
        // loop expr is at least a loop variable, so the step must be
        // const if step is ivar, param, or invalid, all multiplications
        // by indvar are invalid
        return ExprClass::Invalid;

      return ExprClass::IVar;
    }

    auto zero_start = se.getAddRecExpr(se.getConstant(start->getType(), 0),
                                       S->getStepRecurrence(se), S->getLoop(),
                                       S->getNoWrapFlags());

    auto res_expr = visit(zero_start);
    auto start_expr = visit(start);

    return std::max(res_expr, start_expr);
  }

  RetVal visitSequentialUMinExpr(const llvm::SCEVSequentialUMinExpr *S) {
    // todo
    return ExprClass::Invalid;
  }
  RetVal visitUnknown(const llvm::SCEVUnknown *S) {
    const auto value = S->getValue();

    if (context.constants.find(value) != context.constants.end())
      return ExprClass::Constant;

    if (context.parameters.contains(value))
      return ExprClass::Param;

    if (context.getIVarIndex(value) >= 0)
      return ExprClass::IVar;

    if (auto instr = llvm::dyn_cast<llvm::Instruction>(S->getValue())) {
      switch (instr->getOpcode()) {
      case llvm::BinaryOperator::SRem:
        return visitSRemInstruction(instr);
      default:
        break;
      }
    }
    return ExprClass::Invalid;
  }

  RetVal visitSRemInstruction(llvm::Instruction *instr) {
    auto lhs = visit(se.getSCEV(instr->getOperand(0)));
    auto rhs = visit(se.getSCEV(instr->getOperand(1)));

    // only modulo by a constant
    if (rhs != ExprClass::Constant)
      return ExprClass::Invalid;

    return lhs;
  }

  RetVal visitCouldNotCompute(const llvm::SCEVCouldNotCompute *S) {
    // todo
    return ExprClass::Invalid;
  }

  RetVal mergeNary(const llvm::SCEVNAryExpr *S) {
    auto val = visit(S->getOperand(0));

    for (int i = 1; i < S->getNumOperands(); ++i) {
      auto inVal = visit(S->getOperand(i));
      val = std::max(val, inVal);

      if (val == ExprClass::Invalid)
        return ExprClass::Invalid;
    }
    return val;
  }
};

struct PscopDetection::Pimpl {
  using LoopAnalysis = llvm::DenseMap<llvm::Loop *, LoopDetection>;

  llvm::Function &f;
  CudaParameters &cuda;
  SccOrdering &scc;
  ConditionalDominanceAnalysis &cda;
  llvm::RegionInfo &ri;
  llvm::LoopInfo &li;
  llvm::ScalarEvolution &se;

  AffineContext global_context;

  llvm::DenseMap<const llvm::BranchInst *, ConditionalContext>
      conditional_analysis{};
  LoopAnalysis loop_analysis{};

  void initialize() {
    for (auto &arg : f.args())
      global_context.parameters.insert(&arg);

    for (auto &[dim, count] : cuda.getDimCounts()) {
      global_context.induction_vars.emplace_back(InstantiationVariable{
          .kind = (is_grid_dim(dim) ? InstantiationVariable::Kind::Block
                                    : InstantiationVariable::Kind::Thread),
          .lower_bound = 0,
          .upper_bound = count,
          .dim = dim,
      });
    }

    for (auto &bb : f)
      for (auto &instr : bb) {
        llvm::Value *value = &instr;
        if (auto intrin = cuda.getIntrinsic(value)) {
          if (intrin->type == IntrinsicType::id) {
            for (auto &elem : global_context.induction_vars) {
              if (elem.dim == intrin->dim) {
                elem.value = value;
                break;
              }
            }
          } else {
            // type is count
            global_context.constants[value] = cuda.getCount(intrin->dim);
          }
        }
      }
  }

  llvm::Optional<LoopDetection> validateLoop(llvm::Loop *loop,
                                             const AffineContext &context) {
    if (auto bounds = loop->getBounds(se)) {
      auto lower_bound = &bounds->getInitialIVValue();
      auto upper_bound = &bounds->getFinalIVValue();

      // validate bounds
      auto validator = ScevValidator{.se = se, .context = context};
      if (validator.visit(se.getSCEV(lower_bound)) != ExprClass::Invalid &&
          validator.visit(se.getSCEV(upper_bound)) != ExprClass::Invalid) {
        auto val = loop->getInductionVariable(se);
        assert(val);

        auto me = context;
        me.induction_vars.emplace_back(InstantiationVariable{
            .value = val,
            .kind = InstantiationVariable::Kind::Loop,
            .lower_bound = lower_bound,
            .upper_bound = upper_bound,
        });
        return LoopDetection{
            .context = me,
            .affine_loop = loop,
            .is_affine = true,
        };
      }
    }

    return llvm::None;
  }

  LoopAnalysis analyzeLoops() {
    auto analysis = LoopAnalysis{};
    analysis[nullptr] = LoopDetection{
        .context = global_context, .affine_loop = nullptr, .is_affine = true};

    for (auto &loop : li.getLoopsInPreorder()) {
      auto parent_analysis = analysis[loop->getParentLoop()];
      if (auto my_analysis = validateLoop(loop, parent_analysis.context))
        analysis[loop] = std::move(*my_analysis);
      else
        analysis[loop] =
            LoopDetection{.context = parent_analysis.context,
                          .affine_loop = parent_analysis.affine_loop,
                          .is_affine = false};
    }

    return analysis;
  }

  ConditionalContext::AtomInfo validateAtom(llvm::Value *atom,
                                            const AffineContext &context) {
    if (auto as_cmp = llvm::dyn_cast<llvm::CmpInst>(atom)) {
      switch (as_cmp->getPredicate()) {
      case llvm::CmpInst::Predicate::ICMP_EQ:
      case llvm::CmpInst::Predicate::ICMP_NE:
      case llvm::CmpInst::Predicate::ICMP_ULT:
      case llvm::CmpInst::Predicate::ICMP_SLT:
      case llvm::CmpInst::Predicate::ICMP_ULE:
      case llvm::CmpInst::Predicate::ICMP_SLE:
      case llvm::CmpInst::Predicate::ICMP_UGT:
      case llvm::CmpInst::Predicate::ICMP_SGT:
      case llvm::CmpInst::Predicate::ICMP_UGE:
      case llvm::CmpInst::Predicate::ICMP_SGE: {
        auto validator = ScevValidator{
            .se = se,
            .context = context,
        };
        if (validator.visit(se.getSCEV(as_cmp->getOperand(0))) !=
                ExprClass::Invalid &&
            validator.visit(se.getSCEV(as_cmp->getOperand(1))) !=
                ExprClass::Invalid) {
          return ConditionalContext::AtomInfo{.is_affine = true};
        }
      }
      default:
        break;
      }
    }
    return ConditionalContext::AtomInfo{.is_affine = false};
  }

  ConditionalContext validateConditional(llvm::Value *cond,
                                         const AffineContext &context) {
    llvm::MapVector<llvm::Value *, ConditionalContext::AtomInfo> ret;
    for (auto &atom : golly::atomize(cond))
      ret[atom] = validateAtom(atom, context);
    return {ret};
  }

  llvm::DenseMap<const llvm::BranchInst *, ConditionalContext>
  analyzeConstraints(const LoopAnalysis &loop_analysis) {
    llvm::DenseMap<const llvm::BranchInst *, ConditionalContext> analysis;
    for (auto br : cda.getBranches()) {
      auto bb = br->getParent();
      auto loop = li.getLoopFor(bb);
      auto &context = loop_analysis.find(loop)->getSecond().context;

      auto cond = br->getCondition();
      analysis[br] = validateConditional(cond, context);
    }
    return analysis;
  }
};

PscopDetection::PscopDetection(llvm::Function &f,
                               llvm::FunctionAnalysisManager &fam)
    : self{new Pimpl{
          .f = f,
          .cuda = fam.getResult<golly::CudaParameterDetectionPass>(f),
          .scc = fam.getResult<golly::SccOrderingAnalysis>(f),
          .cda = fam.getResult<golly::ConditionalDominanceAnalysisPass>(f),
          .ri = fam.getResult<llvm::RegionInfoAnalysis>(f),
          .li = fam.getResult<llvm::LoopAnalysis>(f),
          .se = fam.getResult<llvm::ScalarEvolutionAnalysis>(f),
      }} {
  self->initialize();
  self->loop_analysis = self->analyzeLoops();
  self->conditional_analysis = self->analyzeConstraints(self->loop_analysis);
}

const AffineContext &PscopDetection::getGlobalContext() const {
  return self->global_context;
}

const LoopDetection *PscopDetection::getLoopInfo(const llvm::Loop *loop) {

  auto itr = self->loop_analysis.find(loop);
  return itr != self->loop_analysis.end() ? &itr->second : nullptr;
}

const ConditionalContext *
PscopDetection::getBranchInfo(const llvm::BranchInst *br) {
  auto itr = self->conditional_analysis.find(br);
  if (itr == self->conditional_analysis.end())
    return nullptr;

  for (auto &elem : itr->second.atoms)
    if (elem.second.is_affine == false)
      return nullptr;

  return &itr->second;
}

PscopDetection::PscopDetection(PscopDetection &&) noexcept = default;
PscopDetection &PscopDetection::operator=(PscopDetection &&) noexcept = default;
PscopDetection::~PscopDetection() noexcept = default;

PscopDetectionPass::Result
golly::PscopDetectionPass::run(llvm::Function &f,
                               llvm::FunctionAnalysisManager &fam) {

  return PscopDetection(f, fam);
}
} // namespace golly
int golly::AffineContext::getIVarIndex(const llvm::Value *ptr) const {
  int i = 0;
  for (auto &iv : induction_vars) {
    if (iv.value == ptr)
      return i;
    ++i;
  }
  return -1;
}
void golly::AffineContext::dump(llvm::raw_ostream &os) const {
  for (auto &iv : induction_vars) {
    os << "iv[" << getIVarIndex(iv.value) << "]: ";
    if (iv.value)
      os << *iv.value;
    else
      os << "(nil)";
    os << "\n";
  }
}