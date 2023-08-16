#include <filesystem>
#include <fmt/format.h>
#include <golly/Analysis/PscopBuilder.h>
#include <golly/Analysis/RaceDetection.h>
#include <golly/Analysis/StatementDetection.h>
#include <golly/Support/isl_llvm.h>
#include <llvm/IR/DebugInfo.h>

namespace golly {
AnalysisKey RaceDetector::Key;

Races RaceDetector::run(Function &f, FunctionAnalysisManager &fam) {
  const auto &pscop = fam.getResult<golly::PscopBuilderPass>(f);

  auto tid_to_stmt_inst = reverse(pscop.distribution_schedule);

  // this is Chatarasi's MHP relation, adapted to accommodate thread pairs

  // enumerate all thread pairs
  auto threads = range(pscop.distribution_schedule);
  auto thread_pairs = universal(threads, threads) - identity(threads);

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
  auto writes_to_accesses =
      apply_range(pscop.write_access_relation, // StmtInst -> Memory
                  reverse(pscop.read_access_relation +
                          pscop.write_access_relation)); // Memory -> StmtInst

  // ensure they are on different threads
  // [S->T] -> [SWriteInst -> TAccessInst]
  auto threads_to_wa = domain_intersect(
      reverse(range_product(apply_range(domain_map(writes_to_accesses),
                                        pscop.distribution_schedule),
                            apply_range(range_map(writes_to_accesses),
                                        pscop.distribution_schedule))),
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

  if (!is_empty(conflicting_syncs)) {
    const auto &stmt_info = fam.getResult<golly::StatementDetectionPass>(f);
    auto conflicting_statements =
        unwrap(range(unwrap(domain(conflicting_syncs))));

    for_each(conflicting_statements, [&](const islpp::map &sampl) -> isl_stat {
      auto write_name = islpp::name(sampl, islpp::dim::in);

      auto write_stmt = stmt_info.getStatement(write_name);
      auto read_name = islpp::name(sampl, islpp::dim::out);
      auto read_stmt = stmt_info.getStatement(read_name);

      assert(write_stmt && read_stmt &&
             "these statements should definitely exist");

      auto clashing_tids = apply_range(
          apply_domain(islpp::union_map{sampl}, pscop.distribution_schedule),
          pscop.distribution_schedule);
      auto single_pair = clean(sample(clashing_tids));

      {
        namespace fs = std::filesystem;
        auto &write_dbg_loc =
            *write_stmt->getDefiningInstruction().getDebugLoc().get();

        auto &acc_dbg_loc =
            *read_stmt->getDefiningInstruction().getDebugLoc().get();

        constexpr auto caret_loc = [](const llvm::DILocation &dbg) -> string {
          constexpr auto canonicalize =
              [](const llvm::DILocation &dbg) -> fs::path {
            return fs::canonical(fs::path{dbg.getDirectory().str()} /
                                 fs::path{dbg.getFilename().str()});
          };

          return fmt::format("{}:{}:{}", canonicalize(dbg).string(),
                             dbg.getLine(), dbg.getColumn());
        };

        llvm::dbgs() << "Race detected between: \n  "
                     << caret_loc(write_dbg_loc) << " on thread "
                     << sample(domain(single_pair)) << " and \n  "
                     << caret_loc(acc_dbg_loc) << " on thread "
                     << sample(range(single_pair)) << "\n";

        llvm::dbgs() << sampl << "\n";
        return isl_stat_ok;
      }
    });
  }

  return {};
}
} // namespace golly