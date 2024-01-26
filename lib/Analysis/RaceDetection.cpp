#include "golly/Analysis/RaceDetection.h"
#include "golly/ADT/PairSet.h"

#include "golly/Analysis/CudaParameterDetection.h"
#include "golly/Analysis/PolyhedralBuilder.h"
#include "golly/Analysis/StatementDetection.h"
#include "golly/Support/isl_llvm.h"

#include "golly/Support/GollyOptions.h"
#include "golly/golly.h"
#include <filesystem>

#include <llvm/Support/CommandLine.h>

llvm::cl::opt<bool> golly_verbose("golly-verbose",
                                  llvm::cl::desc("verbose output"));

namespace golly {
AnalysisKey RaceDetector::Key;

ErrorList RaceDetector::run(Function &f, FunctionAnalysisManager &fam) {

  const auto &pscop = fam.getResult<golly::PolyhedralBuilderPass>(f);
  if (pscop.errors)
    return pscop.errors;

  if (auto opts = RunGollyPass::getOptions(); opts && opts->verboseLog)
    llvm::dbgs() << pscop << "\n";

  auto race_check = may_happen_in_parallel(pscop, pscop.write_access_relation,
                                           pscop.write_access_relation +
                                               pscop.read_access_relation);

  auto atomic_race_check = may_happen_in_parallel(
      pscop, pscop.atomic_access_relation,
      pscop.write_access_relation + pscop.read_access_relation);

  auto conflicting_syncs = clean(race_check + atomic_race_check);

  bool has_conflicts = !is_empty(conflicting_syncs);

  // comment out to reduce strictness
  if (auto opts = RunGollyPass::getOptions(); !opts || opts->strict)
    has_conflicts &= islpp::dims(conflicting_syncs, islpp::dim::param) == 0;
  else
    llvm::dbgs() << "relaxing restrictions";

  if (has_conflicts) {
    const auto &stmt_info = fam.getResult<golly::StatementDetectionPass>(f);

    auto conflicting_statements =
        unwrap(range(unwrap(domain(conflicting_syncs))));

    ErrorList errs;

    PairSet<const Statement *, const Statement *> stmt_pairs;
    auto get_cta =
        apply_range(pscop.thread_allocation,
                    islpp::union_map(pscop.thread_expressions.tau2cta));
    auto same_cta = apply_range(get_cta, reverse(get_cta));
    auto get_warp = chain_apply_range(
        pscop.thread_allocation,
        islpp::union_map(pscop.thread_expressions.tau2warpTuple),
        islpp::union_map(pscop.thread_expressions.warpTuple2warp));
    auto same_warp = apply_range(get_warp, reverse(get_warp));

    for_each(conflicting_statements, [&](const islpp::map &sampl) {
      auto write_stmt =
          stmt_info.getStatement(islpp::name(sampl, islpp::dim::in));
      auto read_stmt =
          stmt_info.getStatement(islpp::name(sampl, islpp::dim::out));

      // don't revisit the same pair
      {
        auto pair = write_stmt < read_stmt ? std::pair{write_stmt, read_stmt}
                                           : std::pair{read_stmt, write_stmt};
        if (stmt_pairs.contains(pair))
          return;
        stmt_pairs.insert(pair);
      }

      assert(write_stmt && read_stmt &&
             "these statements should definitely exist");
      Level conflict_level = Level::Grid;
      {
        // check if filtering out same block statements removes conflicts
        {
          if (is_empty(islpp::union_map{sampl} - same_cta)) {
            conflict_level = Level::Block;

            // we now know all conflicts are block-level
            // check if filtering out same warp statements removes conflicts
            {
              if (is_empty(islpp::union_map{sampl} - same_warp))
                conflict_level = Level::Warp;
            }
          }
        }
      }

      auto clashing_tids = apply_range(
          apply_domain(islpp::union_map{sampl}, pscop.thread_allocation),
          pscop.thread_allocation);
      auto single_pair = clean(sample(clashing_tids));

      errs.emplace_back(DataRace{.instr1 = write_stmt,
                                 .instr2 = read_stmt,
                                 .level = conflict_level,
                                 .clashing_tids = single_pair});
    });

    return errs;
  }
  return {};
}
} // namespace golly