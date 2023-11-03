#ifndef GOLLY_GOLLY_H
#define GOLLY_GOLLY_H

#include <llvm/IR/PassManager.h>
namespace golly {
class GollyOptions;

using llvm::Function;
using llvm::FunctionAnalysisManager;
using llvm::PassInfoMixin;
using llvm::PreservedAnalyses;

class RunGollyPass : public PassInfoMixin<RunGollyPass> {
public:
  RunGollyPass() = default;
  RunGollyPass(GollyOptions &opt);
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &fam);

  static llvm::Optional<GollyOptions> getOptions();
};

} // namespace golly

#endif /* GOLLY_GOLLY_H */
