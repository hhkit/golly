#include <golly/Analysis/DimensionDetection.h>
#include <golly/Analysis/PscopDetection.h>
#include <llvm/Analysis/RegionInfo.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Transforms/Scalar/LoopRotation.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>

llvm::PassPluginLibraryInfo getGollyPluginInfo() {
  using llvm::ArrayRef;
  using llvm::PassBuilder;
  using llvm::StringRef;
  return {
      LLVM_PLUGIN_API_VERSION, "golly", LLVM_VERSION_STRING,
      [](PassBuilder &PB) {
        PB.registerAnalysisRegistrationCallback(
            [](llvm::FunctionAnalysisManager &fam) {
              fam.registerPass([]() { return golly::DimensionDetection(); });
              fam.registerPass([]() { return golly::PscopDetectionPass(); });
            });

        PB.registerPipelineParsingCallback(
            [](StringRef Name, llvm::LoopPassManager &lpm,
               ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
              if (Name == "golly-canonicalize") {
                lpm.addPass(llvm::LoopRotatePass());
                return true;
              }
              return false;
            });

        PB.registerPipelineParsingCallback(
            [](StringRef Name, llvm::FunctionPassManager &PM,
               ArrayRef<llvm::PassBuilder::PipelineElement>) {
              if (Name == "golly-canonicalize") {
                PM.addPass(llvm::PromotePass());
                return true;
              }
              if (Name == "golly") {
                // PM.addPass(golly::PscopDetectionPass());
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
