#include "golly/Analysis/SccOrdering.h"

#include <concepts>
#include <llvm/ADT/SCCIterator.h>
#include <llvm/IR/CFG.h>

namespace golly {
SccOrdering::SccOrdering(llvm::Function &f) {
  auto b = llvm::scc_begin(&f);
  auto e = llvm::scc_end(&f);

  std::vector<const llvm::BasicBlock *> bbs;
  for (auto i = b; i != e; ++i) {
    auto &v = *i;
    for (auto bbi = v.begin(); bbi != v.end(); bbi++)
      bbs.emplace_back(*bbi);
  }

  int counter = 0;
  for (auto itr = bbs.rbegin(); itr != bbs.rend(); ++itr)
    order[*itr] = counter++;
}

int SccOrdering::getPosition(const llvm::BasicBlock *bb) const {
  if (auto itr = order.find(bb); itr != order.end())
    return itr->second;
  return -1;
}

SccOrdering SccOrderingAnalysis::run(llvm::Function &f,
                                     llvm::FunctionAnalysisManager &fam) {
  return SccOrdering(f);
}

const llvm::BasicBlock *golly::SccOrdering::front() {
  return order.front().first;
}

void SccOrdering::sort(std::span<const llvm::BasicBlock *> r) {
  std::ranges::sort(
      r, [&](const llvm::BasicBlock *lhs, const llvm::BasicBlock *rhs) {
        return order[lhs] < order[rhs];
      });
}
} // namespace golly