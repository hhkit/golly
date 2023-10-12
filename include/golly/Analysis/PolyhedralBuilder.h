#ifndef ANALYSIS_POLYHEDRALBUILDER_H
#define ANALYSIS_POLYHEDRALBUILDER_H

#include "golly/Analysis/Pscop.h"

#include <llvm/IR/PassManager.h>

namespace golly {
class PolyhedralBuilderPass
    : public llvm::AnalysisInfoMixin<PolyhedralBuilderPass> {
public:
  using Result = Pscop;
  static inline llvm::AnalysisKey Key;

  Result run(llvm::Function &f, llvm::FunctionAnalysisManager &fam);
};
} // namespace golly
#endif /* ANALYSIS_POLYHEDRALBUILDER_H */
