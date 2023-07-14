#ifndef GOLLY_ANALYSIS_PSCOPDETECTION_H
#define GOLLY_ANALYSIS_PSCOPDETECTION_H

#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>
namespace llvm {
class RegionInfo;
}

namespace golly {

using llvm::Function;
using llvm::FunctionAnalysisManager;
using llvm::FunctionPass;
using llvm::PassInfoMixin;
using llvm::PreservedAnalyses;
using llvm::RegionInfo;

class PscopDetectionPass : public PassInfoMixin<PscopDetectionPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  RegionInfo *RI{};
};
} // namespace golly

#endif /* GOLLY_ANALYSIS_PSCOPDETECTION_H */
