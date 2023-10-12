#ifndef GOLLY_ANALYSIS_PSCOPBUILDER_H
#define GOLLY_ANALYSIS_PSCOPBUILDER_H

#include "golly/Analysis/Pscop.h"
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
using llvm::PreservedAnalyses;

using std::unique_ptr;

class PscopBuilderPass : public AnalysisInfoMixin<PscopBuilderPass> {
public:
  using Result = Pscop;
  static AnalysisKey Key;

  Result run(Function &f, FunctionAnalysisManager &fam);
};

} // namespace golly

#endif /* GOLLY_ANALYSIS_PSCOPBUILDER_H */
