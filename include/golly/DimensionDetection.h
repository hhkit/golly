#ifndef GOLLY_DIMENSIONDETECTION_H
#define GOLLY_DIMENSIONDETECTION_H

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
  static char ID;
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
};
} // namespace golly

#endif /* GOLLY_DIMENSIONDETECTION_H */
