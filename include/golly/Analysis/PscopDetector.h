#ifndef ANALYSIS_PSCOPDETECTOR_H
#define ANALYSIS_PSCOPDETECTOR_H

#include "golly/Analysis/SccOrdering.h"

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Analysis/RegionInfo.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PassManager.h>

#include <concepts>
#include <span>
#include <variant>
namespace golly {

struct InstantiationVariable {
  enum class Kind {
    Thread,
    Loop,
  };

  using Expr = std::variant<llvm::Value *, int>;
  Kind kind;
  Expr lower_bound;
  Expr upper_bound;
};

struct AffineContext {
  llvm::MapVector<const llvm::Value *, InstantiationVariable> induction_vars;
  llvm::DenseSet<llvm::Value *> parameters;
  llvm::MapVector<const llvm::Value *, int> constants;

  int getIndexOfIVar(const llvm::Value *ptr) const;
  void dump(llvm::raw_ostream &os) const;
};

struct LoopDetection {
  AffineContext context;
  llvm::Optional<InstantiationVariable> ivar_introduced;
};

struct ConditionalContext {
  struct AtomInfo {
    bool is_affine = false;
  };
  llvm::MapVector<llvm::Value *, AtomInfo> atoms;
};

class PscopDetection {
public:
  const AffineContext &getGlobalContext() const;
  const LoopDetection *getLoopInfo(const llvm::Loop *);
  const ConditionalContext *getBranchInfo(const llvm::BranchInst *);

  PscopDetection(llvm::Function &f, llvm::FunctionAnalysisManager &fam);
  PscopDetection(PscopDetection &&) noexcept;
  PscopDetection &operator=(PscopDetection &&) noexcept;
  ~PscopDetection() noexcept;

private:
  struct Pimpl;
  std::unique_ptr<Pimpl> self;
};

class PscopDetectionPass : public llvm::AnalysisInfoMixin<PscopDetectionPass> {
public:
  using Result = PscopDetection;
  static inline llvm::AnalysisKey Key;

  Result run(llvm::Function &f, llvm::FunctionAnalysisManager &fam);
};

} // namespace golly

#endif /* ANALYSIS_PSCOPDETECTOR_H */
