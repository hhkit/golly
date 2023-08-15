#include <golly/Analysis/PscopBuilder.h>
#include <golly/Analysis/RaceDetection.h>
#include <golly/Support/isl_llvm.h>

namespace golly {
AnalysisKey RaceDetector::Key;

Races RaceDetector::run(Function &f, FunctionAnalysisManager &fam) {
  const auto &pscop = fam.getResult<golly::PscopBuilderPass>(f);
  llvm::dbgs() << "pscop:\n" << pscop << "\n";

  auto tid_to_stmt_inst = reverse(pscop.distribution_schedule);

  // this is Chatarasi's MHP relation, adapted to accommodate thread pairs

  // enumerate all thread pairs
  auto threads = range(pscop.distribution_schedule);
  auto thread_pairs = unwrap(cross(threads, threads)) - identity(threads);

  // [S -> T] -> StmtInsts of S and T
  auto dmn_insts = apply_range(domain_map(thread_pairs),
                               tid_to_stmt_inst) + // [ S -> T ] -> StmtInsts(S)
                   apply_range(range_map(thread_pairs),
                               tid_to_stmt_inst); // [ S -> T ] -> StmtInsts(T)

  // [[S -> T] -> StmtInst] -> Time
  auto dmn_timings = apply_range(range_map(dmn_insts), pscop.temporal_schedule);

  // [[S -> T] -> StmtInst] -> Time
  auto sync_timings =
      apply_range(range_map(pscop.sync_schedule), pscop.temporal_schedule);

  // [[S->T] -> StmtInst] -> [[S->T] -> SyncInst]
  auto barrs_over_stmts = ([&]() {
    //  [[S -> T] -> StmtInst] -> [[S -> T] -> SyncInst]
    // but we have mismatched S->T
    auto bars_lex_stmts = dmn_timings <<= sync_timings;

    // first, we zip to obtain: [[S->T] -> [S->T]] -> [StmtInst -> SyncInst]
    auto zipped = zip(bars_lex_stmts);

    // then we filter to [[S->T] == [S->T]] -> [StmtInst -> SyncInst]
    auto filtered =
        domain_intersect(zipped, wrap(identity(wrap(thread_pairs))));
    // then we unzip to retrieve the original
    // [[S->T] -> StmtInst] -> [[S->T] -> SyncInst]
    return zip(filtered);
  })();

  // todo: confirm this step does not lose information
  // [[S->T] -> StmtInst] -> SyncTime
  auto sync_times = lexmin(apply_range(range_factor_range(barrs_over_stmts),
                                       pscop.temporal_schedule));

  auto same_range = wrap(identity(range(pscop.temporal_schedule)));

  // WriteInst -> AccessInst on the same variables
  auto writes_to_accesses = apply_range(
      pscop.write_access_relation,
      reverse(pscop.read_access_relation + pscop.write_access_relation));

  // ensure they are on different threads
  // [S->T] -> [SWriteInst -> TAccessInst]
  auto threads_to_wa = domain_intersect(
      reverse(range_product(apply_range(domain_map(writes_to_accesses),
                                        pscop.distribution_schedule),
                            apply_range(range_map(writes_to_accesses),
                                        pscop.distribution_schedule))),
      wrap(thread_pairs));

  // -------------------------------------------------------
  // --------------------------------------------------------
  auto conflicting_syncs = ([&]() {
    // [[S->T] -> [SWriteInst -> TAccessInst]] -> [S->T]
    const auto thread_pairs = domain_map(threads_to_wa);

    // [[S->T] -> [SWriteInst -> TAccessInst]] -> SWriteInst
    // [[S->T] -> [SWriteInst -> TAccessInst]] -> [[S->T] -> SWriteInst]
    // [[S->T] -> [SWriteInst -> TAccessInst]] -> SyncTimeS
    const auto writeinsts = range_factor_domain(range_map(threads_to_wa));
    const auto tiedwrites = range_product(thread_pairs, writeinsts);
    const auto writetimes = apply_range(tiedwrites, sync_times);

    // [[S->T] -> [SWriteInst -> TAccessInst]] -> TAccessInst
    // [[S->T] -> [SWriteInst -> TAccessInst]] -> [[S->T] -> TAccessInst]
    // [[S->T] -> [SWriteInst -> TAccessInst]] -> SyncTimeT
    const auto accinsts = range_factor_range(range_map(threads_to_wa));
    const auto tiedaccs = range_product(thread_pairs, accinsts);
    const auto acctimes = apply_range(tiedaccs, sync_times);

    const auto syncs = range_product(writetimes, acctimes);

    return range_intersect(syncs, same_range);
  })();

  if (!is_empty(conflicting_syncs)) {
    llvm::dbgs() << "race detected\n" << conflicting_syncs << "\n";
  }

  // apply_range(range_factor_domain(domain_map(threads_to_wa)), sync_times)

  // we can only know the next lexicographic sync if we have the two threads
  // synchronizing
  // we can use stmt inst
  // [[S -> T] -> StmtInst] -> sync time
  // maps to the lexmin(StmtInst < sched(S,T))...
  return {};
}
} // namespace golly