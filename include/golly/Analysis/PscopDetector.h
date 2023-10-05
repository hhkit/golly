#ifndef ANALYSIS_PSCOPDETECTOR_H
#define ANALYSIS_PSCOPDETECTOR_H

#include <llvm/IR/PassManager.h>

namespace golly {
struct PscopDetection {
  // Pscop detection region
};

class PscopDetectionPass : public llvm::AnalysisInfoMixin<PscopDetectionPass> {
public:
  using Result = PscopDetection;
  static inline llvm::AnalysisKey Key;

  Result run(llvm::Function &f, llvm::FunctionAnalysisManager &fam);
};

} // namespace golly

#endif /* ANALYSIS_PSCOPDETECTOR_H */
