#ifndef GOLLY_ANALYSIS_STATEMENTDETECTION_H
#define GOLLY_ANALYSIS_STATEMENTDETECTION_H
// #include <llvm/IR/Instruction.h>
#include <golly/Analysis/Statements.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/ilist_node.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/PassManager.h>

#include <list>

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
using llvm::StringMap;

using std::shared_ptr;

// this analysis splits basic blocks into statements
class StatementDetection {
  DenseMap<const BasicBlock *, std::vector<Statement>> map;
  StringMap<const Statement *> cached_names;

public:
  void analyze(const Function &f);
  const Statement *getStatement(string_view name) const;
  std::span<const Statement> iterateStatements(const BasicBlock &bb) const;

  llvm::raw_ostream &dump(llvm::raw_ostream &) const;
};

class StatementDetectionPass
    : public AnalysisInfoMixin<StatementDetectionPass> {
public:
  using Result = StatementDetection;
  static AnalysisKey Key;

  Result run(Function &f, FunctionAnalysisManager &fam);
};

} // namespace golly

#endif /* GOLLY_ANALYSIS_STATEMENTDETECTION_H */
