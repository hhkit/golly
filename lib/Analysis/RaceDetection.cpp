#include "golly/Analysis/RaceDetection.h"
#include "golly/ADT/PairSet.h"
#include "golly/Analysis/CudaParameterDetection.h"
#include "golly/Analysis/PolyhedralBuilder.h"
#include "golly/Analysis/StatementDetection.h"
#include "golly/Support/isl_llvm.h"
#include <filesystem>
#include <fmt/format.h>

#include <llvm/Support/CommandLine.h>

llvm::cl::opt<bool> golly_verbose("golly-verbose",
                                  llvm::cl::desc("verbose output"));

namespace golly {
AnalysisKey RaceDetector::Key;

ErrorList RaceDetector::run(Function &f, FunctionAnalysisManager &fam) {

  const auto &pscop = fam.getResult<golly::PolyhedralBuilderPass>(f);
  if (pscop.errors)
    return pscop.errors;

  if (golly_verbose)
    llvm::dbgs() << pscop << "\n";

  auto tid_to_stmt_inst = reverse(pscop.thread_allocation);

  // this is Chatarasi's MHP relation, adapted to accommodate thread pairs

  auto &sync_times = pscop.sync_schedule;

  // enumerate all thread pairs
  auto threads = range(pscop.thread_allocation);
  auto thread_pairs = universal(threads, threads) - identity(threads);

  // [S -> T] -> StmtInsts of S and T
  auto dmn_insts = apply_range(domain_map(thread_pairs),
                               tid_to_stmt_inst) + // [ S -> T ] -> StmtInsts(S)
                   apply_range(range_map(thread_pairs),
                               tid_to_stmt_inst); // [ S -> T ] -> StmtInsts(T)

  auto same_range = wrap(identity(range(pscop.temporal_schedule)));

  // WriteInst -> AccessInst on the same variables
  auto writes_to_accesses =
      apply_range(pscop.write_access_relation, // StmtInst -> Memory
                  reverse(pscop.read_access_relation +
                          pscop.write_access_relation)); // Memory -> StmtInst

  // ensure they are on different threads
  // [S->T] -> [SWriteInst -> TAccessInst]
  auto threads_to_wa = domain_intersect(
      reverse(range_product(
          apply_range(domain_map(writes_to_accesses), pscop.thread_allocation),
          apply_range(range_map(writes_to_accesses), pscop.thread_allocation))),
      wrap(thread_pairs));

  auto conflicting_syncs = ([&]() {
    // [[S->T] -> [SWriteInst -> TAccessInst]] -> [S->T]
    const auto thread_pairs = domain_map(threads_to_wa);
    // [[S->T] -> [SWriteInst -> TAccessInst]] -> [SWriteInst -> TAccessInst]
    const auto inst_pairs = range_map(threads_to_wa);

    // [[S->T] -> [SWriteInst -> TAccessInst]] -> SWriteInst
    // [[S->T] -> [SWriteInst -> TAccessInst]] -> [[S->T] -> SWriteInst]
    // [[S->T] -> [SWriteInst -> TAccessInst]] -> SyncTimeS
    const auto writeinsts = range_factor_domain(inst_pairs);
    const auto tiedwrites = range_product(thread_pairs, writeinsts);
    const auto writetimes = apply_range(tiedwrites, sync_times);

    // [[S->T] -> [SWriteInst -> TAccessInst]] -> TAccessInst
    // [[S->T] -> [SWriteInst -> TAccessInst]] -> [[S->T] -> TAccessInst]
    // [[S->T] -> [SWriteInst -> TAccessInst]] -> SyncTimeT
    const auto accinsts = range_factor_range(inst_pairs);
    const auto tiedaccs = range_product(thread_pairs, accinsts);
    const auto acctimes = apply_range(tiedaccs, sync_times);

    // [[S->T] -> [SWriteInst -> TAccessInst]] -> [SyncTimeS, SyncTimeT]
    const auto syncs = range_product(writetimes, acctimes);

    return range_intersect(syncs, same_range);
  })();

  conflicting_syncs = clean(conflicting_syncs);

  bool has_conflicts = !is_empty(conflicting_syncs);

  // comment out to reduce strictness
  has_conflicts &= islpp::dims(conflicting_syncs, islpp::dim::param) == 0;

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