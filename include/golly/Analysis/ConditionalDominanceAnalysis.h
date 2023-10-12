#ifndef GOLLY_ANALYSIS_CONDITIONALDOMINANCEANALYSIS_H
#define GOLLY_ANALYSIS_CONDITIONALDOMINANCEANALYSIS_H

#include <memory>
#include <span>

#include <llvm/IR/Instructions.h>
#include <llvm/IR/PassManager.h>

namespace golly {
class ConditionalDominanceAnalysis {
public:
  ConditionalDominanceAnalysis(llvm::Function &f,
                               llvm::FunctionAnalysisManager &fam);

  ConditionalDominanceAnalysis(ConditionalDominanceAnalysis &&rhs) noexcept;
  ConditionalDominanceAnalysis &
  operator=(ConditionalDominanceAnalysis &&rhs) noexcept;
  ~ConditionalDominanceAnalysis() noexcept;

  llvm::ArrayRef<llvm::BasicBlock *>
  getTrueBranch(const llvm::BranchInst *instr);
  llvm::ArrayRef<llvm::BasicBlock *>
  getFalseBranch(const llvm::BranchInst *instr);
  llvm::ArrayRef<const llvm::BranchInst *> getBranches();

private:
  struct Pimpl;
  std::unique_ptr<Pimpl> self;
};

class ConditionalDominanceAnalysisPass
    : public llvm::AnalysisInfoMixin<ConditionalDominanceAnalysisPass> {
public:
  using Result = ConditionalDominanceAnalysis;
  static inline llvm::AnalysisKey Key;
  Result run(llvm::Function &f, llvm::FunctionAnalysisManager &fam);
};
} // namespace golly

#endif /* GOLLY_ANALYSIS_CONDITIONALDOMINANCEANALYSIS_H */
