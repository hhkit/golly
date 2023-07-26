#ifndef GOLLY_ANALYSIS_PSCOPDETECTION_H
#define GOLLY_ANALYSIS_PSCOPDETECTION_H

#include <llvm/ADT/SmallSet.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>
namespace llvm {
class RegionInfo;
}

namespace golly {

using llvm::AnalysisInfoMixin;
using llvm::AnalysisKey;
using llvm::Function;
using llvm::FunctionAnalysisManager;
using llvm::FunctionPass;
using llvm::PassInfoMixin;
using llvm::PreservedAnalyses;
using llvm::RegionInfo;

struct PscopDetectionMap {
  llvm::SmallDenseSet<llvm::Instruction *> isScop;
};

class PscopDetectionPass : public AnalysisInfoMixin<PscopDetectionPass> {
public:
  using Result = PscopDetectionMap;
  static AnalysisKey Key;
  Result run(Function &F, FunctionAnalysisManager &AM);
};

class RunPscopDetection : public PassInfoMixin<RunPscopDetection> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  ;
};
} // namespace golly

#endif /* GOLLY_ANALYSIS_PSCOPDETECTION_H */
