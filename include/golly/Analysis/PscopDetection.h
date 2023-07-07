#ifndef GOLLY_ANALYSIS_PSCOPDETECTION_H
#define GOLLY_ANALYSIS_PSCOPDETECTION_H

#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>

namespace golly {

using llvm::Function;
using llvm::FunctionAnalysisManager;
using llvm::FunctionPass;
using llvm::PassInfoMixin;
using llvm::PreservedAnalyses;

class PscopDetection : public PassInfoMixin<PscopDetection> {
public:
  static char ID;
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
};
} // namespace golly

#endif /* GOLLY_ANALYSIS_PSCOPDETECTION_H */
