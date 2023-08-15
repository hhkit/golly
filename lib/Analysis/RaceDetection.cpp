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
  // [S->T] -> [StmtInst -> Time]
  auto dmn_timings = apply_range(range_map(dmn_insts), pscop.temporal_schedule);

  isl_union_map_lex_ge_at_multi_union_pw_aff;
  isl_union_map_lex_lt_union_map;

  if (1) // test bed
  {
    auto val = islpp::union_map{"{ [tidx] -> [tidx, 0] }"}
               << islpp::union_map{"{ [tidx] -> [tidx, 1] }"};
    llvm::dbgs() << "test:" << val << "\n";

    llvm::dbgs() << thread_pairs << "\n";
  }
  // we can only know the next lexicographic sync if we have the two threads
  // synchronizing
  // we can use stmt inst
  // [[S -> T] -> StmtInst] -> sync time
  // maps to the lexmin(StmtInst < sched(S,T))...

  // llvm::dbgs() << (dmn_insts <<= pscop.sync_schedule) << "\n";
  return {};
}
} // namespace golly