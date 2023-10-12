#include "golly/Analysis/ConditionalDominanceAnalysis.h"
#include "golly/Analysis/SccOrdering.h"
#include <llvm/ADT/SetVector.h>
#include <llvm/Analysis/IteratedDominanceFrontier.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Dominators.h>

namespace golly {

struct ConditionalDominanceAnalysis::Pimpl {
  struct DominanceDomains {
    llvm::SetVector<llvm::BasicBlock *> true_blocks;
    llvm::SetVector<llvm::BasicBlock *> false_blocks;
  };

  llvm::DenseMap<const llvm::BranchInst *, DominanceDomains> map;
  std::vector<llvm::BranchInst *> branches;
};

ConditionalDominanceAnalysis::ConditionalDominanceAnalysis(
    llvm::Function &f, llvm::FunctionAnalysisManager &fam)
    : self{new Pimpl} {
  auto &dom_tree = fam.getResult<llvm::DominatorTreeAnalysis>(f);
  auto &pdt = fam.getResult<llvm::PostDominatorTreeAnalysis>(f);
  auto &scc = fam.getResult<golly::SccOrderingAnalysis>(f);
  auto &map = self->map;
  auto branch_lut =
      llvm::DenseMap<const llvm::BasicBlock *, llvm::BranchInst *>{};
  auto &branches = self->branches = [&f, &branch_lut]() {
    std::vector<llvm::BranchInst *> branches;
    for (auto &bb : f)
      for (auto &instr : bb)
        if (auto branch = llvm::dyn_cast<llvm::BranchInst>(&instr))
          if (branch->isConditional()) {
            branches.emplace_back(branch);
            branch_lut[&bb] = branch;
          }
    return branches;
  }();

  for (auto &bb : f) {
    llvm::SmallVector<llvm::BasicBlock *> pdf;
    llvm::SmallPtrSet<llvm::BasicBlock *, 1> checkMe{&bb};
    llvm::ReverseIDFCalculator calc{pdt};
    calc.setDefiningBlocks(checkMe);
    calc.calculate(pdf);
    for (auto &pd : pdf) {
      // bb is conditionally dependent on all of pd
      if (auto br = branch_lut.lookup(pd)) {
        const auto true_branch = br->getSuccessor(0);
        const auto false_branch = br->getSuccessor(1);

        auto &domain = map[br];
        if (dom_tree.dominates(true_branch, &bb))
          domain.true_blocks.insert(&bb);

        if (dom_tree.dominates(false_branch, &bb))
          domain.false_blocks.insert(&bb);
      }
    }
  }
}

ConditionalDominanceAnalysis::ConditionalDominanceAnalysis(
    ConditionalDominanceAnalysis &&rhs) noexcept = default;
ConditionalDominanceAnalysis &ConditionalDominanceAnalysis::operator=(
    ConditionalDominanceAnalysis &&rhs) noexcept = default;
ConditionalDominanceAnalysis::~ConditionalDominanceAnalysis() noexcept =
    default;

llvm::ArrayRef<llvm::BasicBlock *>
ConditionalDominanceAnalysis::getTrueBranch(const llvm::BranchInst *instr) {
  auto &map = self->map;
  auto itr = map.find(instr);
  if (itr != map.end())
    return itr->second.true_blocks.getArrayRef();
  return {};
}

llvm::ArrayRef<llvm::BasicBlock *>
ConditionalDominanceAnalysis::getFalseBranch(const llvm::BranchInst *instr) {
  auto &map = self->map;
  auto itr = map.find(instr);
  if (itr != map.end())
    return itr->second.false_blocks.getArrayRef();
  return {};
}

llvm::ArrayRef<const llvm::BranchInst *>
ConditionalDominanceAnalysis::getBranches() {
  return self->branches;
}

ConditionalDominanceAnalysisPass::Result
ConditionalDominanceAnalysisPass::run(llvm::Function &f,
                                      llvm::FunctionAnalysisManager &fam) {
  return Result(f, fam);
}
} // namespace golly