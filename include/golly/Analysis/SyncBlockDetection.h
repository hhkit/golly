#ifndef GOLLY_ANALYSIS_SYNCBLOCKDETECTION_H
#define GOLLY_ANALYSIS_SYNCBLOCKDETECTION_H
// #include <llvm/IR/Instruction.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/ilist_node.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/SymbolTableListTraits.h>

#include <memory>
namespace llvm {
class Instruction;
class Function;
class raw_ostream;
} // namespace llvm

namespace golly {
using llvm::AnalysisInfoMixin;
using llvm::AnalysisKey;
using llvm::BasicBlock;
using llvm::DenseMap;
using llvm::Function;
using llvm::FunctionAnalysisManager;
using llvm::ilist_node_with_parent;
using llvm::Instruction;
using llvm::SmallVector;
using llvm::SymbolTableList;

using std::shared_ptr;
using std::unique_ptr;

class SyncBlock {
public:
  using InstListType = SymbolTableList<Instruction>;
  using const_iterator = InstListType::const_iterator;

  SyncBlock(const_iterator b, const_iterator e);

  void addSuccessor(unique_ptr<SyncBlock> child);
  SyncBlock *getSuccessor() const;

  // phase ends in a barrier or return statement
  bool willSynchronize() const;

  const_iterator begin() const { return beg_; }
  const_iterator end() const { return end_; }

private:
  const_iterator beg_, end_, last_;
  unique_ptr<SyncBlock> successor;
};

// this analysis splits basic blocks by the barrier
class SyncBlockDetection {
  DenseMap<const BasicBlock *, unique_ptr<SyncBlock>> map;

public:
  void analyze(const Function &f);
  SyncBlock *getTopLevelPhase(const BasicBlock &f) const;

  llvm::raw_ostream &dump(llvm::raw_ostream &) const;
};

class SyncBlockDetectionPass
    : public AnalysisInfoMixin<SyncBlockDetectionPass> {
public:
  using Result = SyncBlockDetection;
  static AnalysisKey Key;

  Result run(Function &f, FunctionAnalysisManager &fam);
};

} // namespace golly

#endif /* GOLLY_ANALYSIS_SYNCBLOCKDETECTION_H */
