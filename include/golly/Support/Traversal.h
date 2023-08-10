#ifndef GOLLY_SUPPORT_TRAVERSAL_H
#define GOLLY_SUPPORT_TRAVERSAL_H
#include <llvm/ADT/DenseSet.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <queue>

namespace golly {
template <typename T> void bfs(llvm::BasicBlock *first, T &&fn) {
  std::queue<llvm::BasicBlock *> queue;
  llvm::DenseSet<llvm::BasicBlock *> visited;
  queue.push(first);
  visited.insert(first);

  while (!queue.empty()) {
    auto visit = queue.front();
    queue.pop();

    fn(visit);

    for (auto &&child : llvm::successors(visit)) {
      if (!visited.contains(child)) {
        queue.push(child);
        visited.insert(child);
      }
    }
  }
}
} // namespace golly
#endif /* GOLLY_SUPPORT_TRAVERSAL_H */
