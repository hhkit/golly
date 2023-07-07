#include <golly/PscopDetection.h>
#include <llvm/IR/Function.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <isl/cpp.h>
#include <iostream>
namespace golly {
char PscopDetection::ID = 0;

PreservedAnalyses PscopDetection::run(Function &f,
                                      FunctionAnalysisManager &am) {

  std::cerr << "analysis: " << f.getName().str() << std::endl;
  
  // verify the function is 
  // verify the grid of the function

  if (f.isIntrinsic())
    llvm::errs() << f.getName() << " is intrinsic\n";
  return PreservedAnalyses::all();
}
} // namespace golly

llvm::PassPluginLibraryInfo getGollyPluginInfo() {
  using llvm::ArrayRef;
  using llvm::PassBuilder;
  using llvm::StringRef;
  return {LLVM_PLUGIN_API_VERSION, "golly", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::FunctionPassManager &PM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "golly") {
                    PM.addPass(golly::PscopDetection());
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
