#ifndef GOLLY_ANALYSIS_DIMENSIONDETECTION_H
#define GOLLY_ANALYSIS_DIMENSIONDETECTION_H

#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>

namespace golly {

using llvm::AnalysisInfoMixin;
using llvm::AnalysisKey;
using llvm::AnalysisUsage;
using llvm::Function;
using llvm::FunctionAnalysisManager;
using llvm::FunctionPass;
using llvm::PreservedAnalyses;

struct DetectedIntrinsics {
  bool ctaX{}, ctaY{}, ctaZ{};
  bool nctaX{}, nctaY{}, nctaZ{};
  bool tidX{}, tidY{}, tidZ{};
  bool ntidX{}, ntidY{}, ntidZ{};
};

static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                            const DetectedIntrinsics &di) {
  os << "ctaid:" << di.ctaX << di.ctaY << di.ctaZ << " nctaid:" << di.nctaX
     << di.nctaY << di.nctaZ << " ";
  os << "tid:" << di.tidX << di.tidY << di.tidZ << " ntid:" << di.ntidX
     << di.ntidY << di.ntidZ << " ";
  return os;
}

class DimensionDetection : public AnalysisInfoMixin<DimensionDetection> {
public:
  using Result = DetectedIntrinsics;
  static AnalysisKey Key;

  Result run(Function &F, FunctionAnalysisManager &AM);

private:
};

class DimensionDetectionPass : public DimensionDetection {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
} // namespace golly

#endif /* GOLLY_ANALYSIS_DIMENSIONDETECTION_H */
