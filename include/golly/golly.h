#ifndef GOLLY_GOLLY_H
#define GOLLY_GOLLY_H

#include <llvm/IR/PassManager.h>
namespace golly {

using llvm::Function;
using llvm::FunctionAnalysisManager;
using llvm::PassInfoMixin;
using llvm::PreservedAnalyses;

class RunGollyPass : public PassInfoMixin<RunGollyPass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &fam);
};

} // namespace golly

#endif /* GOLLY_GOLLY_H */
