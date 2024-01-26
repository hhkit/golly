#include "golly/Analysis/Pscop.h"
#include "golly/Support/isl_llvm.h"

namespace golly {
islpp::union_map may_happen_in_parallel(const Pscop &pscop,
                                        islpp::union_map lhs,
                                        islpp::union_map rhs) {
  auto same_time = wrap(identity(range(pscop.temporal_schedule)));
  auto tid_to_stmt_inst = reverse(pscop.thread_allocation);
  auto &sync_times = pscop.sync_schedule;
  // enumerate all thread pairs
  auto threads = range(pscop.thread_allocation);
  auto thread_pairs = universal(threads, threads) - identity(threads);

  // [S -> T] -> StmtInsts of S and T
  auto dmn_insts = apply_range(domain_map(thread_pairs),
                               tid_to_stmt_inst) + // [ S -> T ] -> StmtInsts(S)
                   apply_range(range_map(thread_pairs),
                               tid_to_stmt_inst); // [ S -> T ] -> StmtInsts(T)

  // WriteInst -> AccessInst on the same variables
  auto lhs_to_rhs = apply_range(lhs,           // StmtInst -> Memory
                                reverse(rhs)); // Memory -> StmtInst

  // ensure they are on different threads
  // [S->T] -> [SWriteInst -> TAccessInst]
  auto threads_to_wa = domain_intersect(
      reverse(range_product(
          apply_range(domain_map(lhs_to_rhs), pscop.thread_allocation),
          apply_range(range_map(lhs_to_rhs), pscop.thread_allocation))),
      wrap(thread_pairs));

  // [[S->T] -> [SWriteInst -> TAccessInst]] -> [S->T]
  const auto thread_pairs_truple = domain_map(threads_to_wa);
  // [[S->T] -> [SWriteInst -> TAccessInst]] -> [SWriteInst -> TAccessInst]
  const auto inst_pairs = range_map(threads_to_wa);

  // [[S->T] -> [SWriteInst -> TAccessInst]] -> SWriteInst
  // [[S->T] -> [SWriteInst -> TAccessInst]] -> [[S->T] -> SWriteInst]
  // [[S->T] -> [SWriteInst -> TAccessInst]] -> SyncTimeS
  const auto writeinsts = range_factor_domain(inst_pairs);
  const auto tiedwrites = range_product(thread_pairs_truple, writeinsts);
  const auto writetimes = apply_range(tiedwrites, sync_times);

  // [[S->T] -> [SWriteInst -> TAccessInst]] -> TAccessInst
  // [[S->T] -> [SWriteInst -> TAccessInst]] -> [[S->T] -> TAccessInst]
  // [[S->T] -> [SWriteInst -> TAccessInst]] -> SyncTimeT
  const auto accinsts = range_factor_range(inst_pairs);
  const auto tiedaccs = range_product(thread_pairs_truple, accinsts);
  const auto acctimes = apply_range(tiedaccs, sync_times);

  // [[S->T] -> [SWriteInst -> TAccessInst]] -> [SyncTimeS, SyncTimeT]
  const auto syncs = range_product(writetimes, acctimes);

  return range_intersect(syncs, same_time);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Pscop &pscop) {
  os << "domain:\n  " << pscop.instantiation_domain << "\n";
  os << "thread_allocation:\n  " << pscop.thread_allocation << "\n";
  os << "temporal_schedule:\n  " << pscop.temporal_schedule << "\n";
  os << "sync_schedule:\n  " << pscop.sync_schedule << "\n";
  os << "writes:\n  " << pscop.write_access_relation << "\n";
  os << "reads:\n  " << pscop.read_access_relation << "\n";
  os << "atomic accesses: \n" << pscop.atomic_access_relation << "\n";
  return os;
}
} // namespace golly