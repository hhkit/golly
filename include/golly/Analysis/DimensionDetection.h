#ifndef GOLLY_ANALYSIS_DIMENSIONDETECTION_H
#define GOLLY_ANALYSIS_DIMENSIONDETECTION_H

#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>

namespace golly {

using llvm::Function;
using llvm::FunctionAnalysisManager;
using llvm::FunctionPass;
using llvm::PassInfoMixin;
using llvm::PreservedAnalyses;

class DimensionDetection : public PassInfoMixin<DimensionDetection> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
};
} // namespace golly

#endif /* GOLLY_ANALYSIS_DIMENSIONDETECTION_H */
