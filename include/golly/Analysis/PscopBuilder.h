#ifndef GOLLY_ANALYSIS_PSCOPBUILDER_H
#define GOLLY_ANALYSIS_PSCOPBUILDER_H

#include <isl/cpp.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/PassManager.h>
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
using llvm::Region;

using std::unique_ptr;

struct Pscop {
  isl::set iteration_domain;    // a polytope representing the iterations of the
                                // statement
  isl::set distribution_domain; // a polytope representing the parallelism of
                                // the statement

  isl::map schedule; // maps a statement instance in the iteration polytope to a
                     // logical time
  isl::map phase_schedule; // maps a statement instance in the iteration
                           // polytope to a phase point

  isl::map write_access_relation; // maps a statement instance to the set of
                                  // writes it performs
  isl::map read_access_relation;  // maps a statement instance to the set of
                                  // reads it performs

  // dependence relation irrelevant for race detection
};

struct PscopsDetected {};

} // namespace golly

#endif /* GOLLY_ANALYSIS_PSCOPBUILDER_H */
