#ifndef GOLLY_ANALYSIS_PSCOPBUILDER_H
#define GOLLY_ANALYSIS_PSCOPBUILDER_H

#include <golly/Support/isl.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace llvm {
class Function;
} // namespace llvm

namespace golly {

using llvm::AnalysisInfoMixin;
using llvm::AnalysisKey;
using llvm::Function;
using llvm::FunctionAnalysisManager;

using std::unique_ptr;
/*
a note on the type annotations:
  Stmt:
    A statement - an atomic subdivision of instructions that we care about

  StmtInst:
    An instance of a statement, duplicated by parallelism or loops.
    eg. StmtA[0, 1] \in { StmtA[tid, i] : 0 <= tid <= 15 and 0 <= i <= 2}

  tid:
    Thread ID

  Time:
    A multidimensional vector, the lexicographic ordering of which corresponds
    to temporal dependence within a thread

  Access:
    A single dimensional vector that represents an access of a memory location
*/
struct Pscop {
  islpp::union_map instantiation_domain;  // { Stmt -> StmtInst }
  islpp::union_map distribution_schedule; // { StmtInst -> tid }
  islpp::union_map temporal_schedule;     // { StmtInst -> Time }

  islpp::union_map phase_schedule; // tid -> StmtInst

  islpp::union_map write_access_relation; // param -> { StmtInst -> Access }
  islpp::union_map read_access_relation;  // param -> { StmtInst -> Access }

  // dependence relation irrelevant for race detection

  void dump(llvm::raw_ostream &) const;
};

class PscopBuilderPass : public AnalysisInfoMixin<PscopBuilderPass> {
public:
  using Result = Pscop;
  static AnalysisKey Key;

  Result run(Function &f, FunctionAnalysisManager &fam);
};

} // namespace golly

#endif /* GOLLY_ANALYSIS_PSCOPBUILDER_H */
