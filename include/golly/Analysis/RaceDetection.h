#ifndef GOLLY_ANALYSIS_RACEDETECTION_H
#define GOLLY_ANALYSIS_RACEDETECTION_H
#include "golly/ErrorHandling/Error.h"
#include <llvm/IR/PassManager.h>

namespace golly {
using llvm::AnalysisInfoMixin;
using llvm::AnalysisKey;
using llvm::Function;
using llvm::FunctionAnalysisManager;

class RaceDetector : public AnalysisInfoMixin<RaceDetector> {
public:
  using Result = ErrorList;
  static AnalysisKey Key;

  Result run(Function &f, FunctionAnalysisManager &fam);
};
} // namespace golly

#endif /* GOLLY_ANALYSIS_RACEDETECTION_H */
