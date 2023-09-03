#ifndef GOLLY_SUPPORT_TRAVERSAL_H
#define GOLLY_SUPPORT_TRAVERSAL_H
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SCCIterator.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <ranges>
#include <vector>

namespace golly {
template <typename T> void bfs(llvm::Function &f, T &&fn) {

  std::vector<llvm::BasicBlock *> visitus;
  for (auto b = llvm::scc_begin(&f); b != llvm::scc_end(&f); ++b) {
    auto &i = *b;
    for (auto &bb : i)
      visitus.emplace_back(bb);
  }

  for (auto itr = visitus.rbegin(); itr != visitus.rend(); ++itr) {
    fn(*itr);
  }
}
} // namespace golly
#endif /* GOLLY_SUPPORT_TRAVERSAL_H */
