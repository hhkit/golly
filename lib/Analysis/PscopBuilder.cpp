#include <golly/Analysis/PscopBuilder.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/PriorityQueue.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/IR/Instructions.h>
#include <queue>

namespace golly {

namespace detail {
bool isInductionVariable(
    const llvm::Instruction *checkme,
    const llvm::DenseSet<const llvm::Instruction *> &iteration_vars,
    const llvm::StringSet</*nani???*/> &distribution_vars) {
  if (iteration_vars.contains(checkme))
    return true;

  if (auto callinst = llvm::dyn_cast_or_null<llvm::CallInst>(checkme)) {
    const auto fn_name = callinst->getFunction()->getName();
    return distribution_vars.contains(fn_name);
  }

  return false;
}
} // namespace detail

class PscopBuilder {
public:
  void build(const Function &f) {
    // detect allocation domain of function

    // iterate over bbs of function in BFS
    const auto first = &f.getEntryBlock();
    std::queue<const llvm::BasicBlock *> queue;
    queue.emplace(first);
    llvm::DenseSet<const llvm::BasicBlock *> visited{first};

    while (!queue.empty()) {
      const auto visit = queue.front();
      queue.pop();
      // extract iteration domain from enclosing region
      // note:
      // by induction, we will have subsequent enclosing regions'
      // iteration domains as well

      // check our region's constraints
      // are we a loop?
      // if so, add the iteration variable into our iteration domain

      // are we a result of a branch?
      // if so, add the constraint into our set

      // induction variables are a union of allocation vars and iteration vars

      // now we generate our access relations

      // then we enqueue our children if they are not yet enqueued

      for (auto &&elem : llvm::successors(visit)) {
        if (visited.contains(elem))
          continue;

        queue.push(elem);
        visited.insert(elem);
      }
    }
  }

private:
  DenseMap<const Region *, unique_ptr<isl::set>> iteration_domain;
};
} // namespace golly