#ifndef GOLLY_GOLLY_H
#define GOLLY_GOLLY_H

#include <llvm/IR/PassManager.h>
namespace golly {

using llvm::PassInfoMixin;

class RunGollyPass : public PassInfoMixin<RunGollyPass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &fam);
};

} // namespace golly

#endif /* GOLLY_GOLLY_H */
