#ifndef ANALYSIS_SCCORDERING_H
#define ANALYSIS_SCCORDERING_H

#include <ranges>
#include <span>

#include <llvm/ADT/MapVector.h>
#include <llvm/IR/PassManager.h>
namespace golly {

class SccOrdering {
public:
  explicit SccOrdering(llvm::Function &f);
  int getPosition(const llvm::BasicBlock *bb) const;
  void traverse(std::invocable<const llvm::BasicBlock *> auto &&f) const;
  void sort(std::span<const llvm::BasicBlock *> r);
  const llvm::BasicBlock *front();

private:
  llvm::MapVector<const llvm::BasicBlock *, int> order;
};

class SccOrderingAnalysis
    : public llvm::AnalysisInfoMixin<SccOrderingAnalysis> {
public:
  using Result = SccOrdering;
  static inline llvm::AnalysisKey Key;
  Result run(llvm::Function &f, llvm::FunctionAnalysisManager &fam);
};
} // namespace golly

void golly::SccOrdering::traverse(
    std::invocable<const llvm::BasicBlock *> auto &&f) const {
  for (auto &[bb, count] : order)
    if constexpr (std::is_same_v<std::invoke_result_t<decltype(f),
                                                      const llvm::BasicBlock *>,
                                 bool>) {
      if (f(bb))
        return;
    } else
      f(bb);
}

#endif /* ANALYSIS_SCCORDERING_H */
