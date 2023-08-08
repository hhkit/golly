#ifndef GOLLY_ANALYSIS_PSCOPBUILDER_H
#define GOLLY_ANALYSIS_PSCOPBUILDER_H

#include <golly/Support/isl.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace llvm {
class Region;
class Function;
class RegionInfo;
class ScalarEvolution;
} // namespace llvm

namespace golly {

using llvm::AnalysisInfoMixin;
using llvm::AnalysisKey;
using llvm::DenseMap;
using llvm::Function;
using llvm::FunctionAnalysisManager;
using llvm::Region;

using std::unique_ptr;

struct Pscop {
  islpp::union_map
      instantiation_domain; // a mapping of statement to its statement instances
  islpp::union_map temporal_schedule; // a mapping of statement instance to time

  islpp::union_map phase_schedule; // maps a statement instance in the iteration
                                   // polytope to a phase point

  islpp::union_map write_access_relation; // maps a statement instance to the
                                          // set of writes it performs
  islpp::union_map read_access_relation; // maps a statement instance to the set
                                         // of reads it performs

  // dependence relation irrelevant for race detection

  void dump(llvm::raw_ostream &) const;
};

class PscopBuilderPass : public AnalysisInfoMixin<PscopBuilderPass> {
public:
  using Result = Pscop;
  static AnalysisKey Key;

  Result run(Function &f, FunctionAnalysisManager &fam);
};

} // namespace golly

#endif /* GOLLY_ANALYSIS_PSCOPBUILDER_H */
