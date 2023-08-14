#ifndef GOLLY_ANALYSIS_RACEDETECTION_H
#define GOLLY_ANALYSIS_RACEDETECTION_H
#include <llvm/IR/PassManager.h>

namespace golly {
using llvm::AnalysisInfoMixin;
using llvm::AnalysisKey;
using llvm::Function;
using llvm::FunctionAnalysisManager;

struct Races {};

class RaceDetector : public AnalysisInfoMixin<RaceDetector> {
public:
  using Result = Races;
  static AnalysisKey Key;

  Result run(Function &f, FunctionAnalysisManager &fam);
};
} // namespace golly

#endif /* GOLLY_ANALYSIS_RACEDETECTION_H */
