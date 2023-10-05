#include "golly/Analysis/PscopDetector.h"

#include <llvm/Analysis/RegionInfo.h>

namespace golly {
PscopDetectionPass::Result
golly::PscopDetectionPass::run(llvm::Function &f,
                               llvm::FunctionAnalysisManager &fam) {
  auto &ri = fam.getResult<llvm::RegionInfoAnalysis>(f);
  ri.dump();
  return Result();
}
} // namespace golly