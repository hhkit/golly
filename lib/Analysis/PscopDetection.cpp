#include <golly/Analysis/PscopDetection.h>
#include <iostream>
#include <isl/cpp.h>
#include <llvm/IR/Function.h>
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
