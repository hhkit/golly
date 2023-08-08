#include <golly/Analysis/DimensionDetection.h>
#include <golly/Analysis/PscopBuilder.h>
#include <golly/Analysis/PscopDetection.h>
#include <golly/Analysis/SyncBlockDetection.h>
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
              fam.registerPass(
                  []() { return golly::StatementDetectionPass(); });
              fam.registerPass([]() { return golly::PscopBuilderPass(); });
            });

        PB.registerPipelineParsingCallback(
            [](StringRef Name, llvm::FunctionPassManager &PM,
               ArrayRef<llvm::PassBuilder::PipelineElement>) {
              if (Name == "golly") {
                PM.addPass(golly::RunPscopDetection());
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
