#include "golly/golly.h"
#include "golly/Analysis/ConditionalDominanceAnalysis.h"
#include "golly/Analysis/CudaParameterDetection.h"
#include "golly/Analysis/PolyhedralBuilder.h"
#include "golly/Analysis/PscopBuilder.h"
#include "golly/Analysis/PscopDetector.h"
#include "golly/Analysis/RaceDetection.h"
#include "golly/Analysis/SccOrdering.h"
#include "golly/Analysis/StatementDetection.h"

#include <llvm/Analysis/RegionInfo.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Transforms/Scalar/LoopRotation.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>

namespace golly {
PreservedAnalyses RunGollyPass::run(Function &f, FunctionAnalysisManager &fam) {
  if (f.getName() == "_Z10__syncwarpj") {
    return PreservedAnalyses::none();
  }
  fam.getResult<golly::RaceDetector>(f);
  // llvm::outs() < < < < "\n";
  return PreservedAnalyses::all();
}
} // namespace golly

llvm::PassPluginLibraryInfo getGollyPluginInfo() {
  using llvm::ArrayRef;
  using llvm::PassBuilder;
  using llvm::StringRef;

  return {
      LLVM_PLUGIN_API_VERSION, "golly", LLVM_VERSION_STRING,
      [](PassBuilder &PB) {
        PB.registerAnalysisRegistrationCallback(
            [](llvm::FunctionAnalysisManager &fam) {
              fam.registerPass([]() { return golly::PolyhedralBuilderPass(); });
              fam.registerPass(
                  []() { return golly::ConditionalDominanceAnalysisPass(); });
              fam.registerPass([]() { return golly::SccOrderingAnalysis(); });
              fam.registerPass(
                  []() { return golly::CudaParameterDetection(); });
              fam.registerPass(
                  []() { return golly::StatementDetectionPass(); });
              fam.registerPass([]() { return golly::PscopBuilderPass(); });
              fam.registerPass([]() { return golly::PscopDetectionPass(); });
              fam.registerPass([]() { return golly::RaceDetector(); });
            });

        PB.registerPipelineParsingCallback(
            [](StringRef Name, llvm::FunctionPassManager &PM,
               ArrayRef<llvm::PassBuilder::PipelineElement>) {
              if (Name == "golly") {
                PM.addPass(golly::RunGollyPass());
                return true;
              }
              return false;
            });
      }};
}

#ifndef LLVM_GOLLY_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getGollyPluginInfo();
}
#endif
