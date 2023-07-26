#ifndef GOLLY_ANALYSIS_DIMENSIONDETECTION_H
#define GOLLY_ANALYSIS_DIMENSIONDETECTION_H

#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
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
using llvm::SmallDenseMap;
using llvm::SmallVector;

enum class Intrinsic {
  tidX,
  tidY,
  tidZ,
  nTidX,
  nTidY,
  nTidZ,
  ctaX,
  ctaY,
  ctaZ,
  nCtaX,
  nCtaY,
  nCtaZ,
};

const static inline SmallVector<Intrinsic> threadIds{
    Intrinsic::tidX, Intrinsic::tidY, Intrinsic::tidZ,
    Intrinsic::ctaX, Intrinsic::ctaY, Intrinsic::ctaZ};
const static inline SmallVector<Intrinsic> threadDims{
    Intrinsic::nTidX, Intrinsic::nTidY, Intrinsic::nTidZ,
    Intrinsic::nCtaX, Intrinsic::nCtaY, Intrinsic::nCtaZ};

struct DetectedIntrinsics {
  SmallDenseMap<llvm::Value *, Intrinsic> detections;
};

static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                            const DetectedIntrinsics &di) {
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
