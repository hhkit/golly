#include <llvm/Analysis/RegionInfo.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

#include <golly/Analysis/DimensionDetection.h>
#include <golly/Analysis/PscopDetection.h>

llvm::PassPluginLibraryInfo getGollyPluginInfo() {
  using llvm::ArrayRef;
  using llvm::PassBuilder;
  using llvm::StringRef;
  return {LLVM_PLUGIN_API_VERSION, "golly", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerAnalysisRegistrationCallback(
                [](llvm::FunctionAnalysisManager &fam) {
                  fam.registerPass(
                      []() { return golly::DimensionDetection(); });
                });
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::FunctionPassManager &PM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "golly") {
                    (golly::DimensionDetection());
                    PM.addPass(golly::PscopDetectionPass());
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
