#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <golly/Analysis/CudaParameterDetection.h>
#include <golly/Analysis/PolyhedralBuilder.h>
#include <golly/Analysis/PscopBuilder.h>
#include <golly/Analysis/RaceDetection.h>
#include <golly/Analysis/StatementDetection.h>
#include <golly/Support/isl_llvm.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/DebugInfo.h>

llvm::cl::opt<bool> golly_verbose("golly-verbose",
                                  llvm::cl::desc("verbose output"));

llvm::cl::opt<std::string> out_file("golly-out", llvm::cl::desc("output file"));

namespace golly {
AnalysisKey RaceDetector::Key;

Races RaceDetector::run(Function &f, FunctionAnalysisManager &fam) {
  const auto &params = fam.getResult<golly::CudaParameterDetection>(f);
  llvm::outs() << "Running race detection of "
               << llvm::demangle(f.getName().str())
               << " with launch parameters: " << params << "\n";

  const auto &pscop = fam.getResult<golly::PolyhedralBuilderPass>(f);
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

  if (!is_empty(conflicting_syncs) &&
      islpp::dims(conflicting_syncs, islpp::dim::param) == 0) {
    const auto &stmt_info = fam.getResult<golly::StatementDetectionPass>(f);

    auto conflicting_statements =
        unwrap(range(unwrap(domain(conflicting_syncs))));

    auto fstream = std::ofstream(out_file);

    for_each(conflicting_statements, [&](const islpp::map &sampl) {
      auto write_name = islpp::name(sampl, islpp::dim::in);

      auto write_stmt = stmt_info.getStatement(write_name);
      auto read_name = islpp::name(sampl, islpp::dim::out);
      auto read_stmt = stmt_info.getStatement(read_name);

      assert(write_stmt && read_stmt &&
             "these statements should definitely exist");

      auto clashing_tids = apply_range(
          apply_domain(islpp::union_map{sampl}, pscop.thread_allocation),
          pscop.thread_allocation);
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

        llvm::errs() << "Race detected in " << llvm::demangle(f.getName().str())
                     << " between: \n  " << caret_loc(write_dbg_loc)
                     << " on thread " << sample(domain(single_pair))
                     << " and \n  " << caret_loc(acc_dbg_loc) << " on thread "
                     << sample(range(single_pair)) << "\n";

        llvm::errs() << sampl << "\n";

        if (fstream) {
          fstream << caret_loc(write_dbg_loc) << "\n";
          fstream << caret_loc(acc_dbg_loc) << "\n";
          fstream.close();
        }
      }
    });
  } else {
    llvm::outs() << "No race detected in " << llvm::demangle(f.getName().str())
                 << "\n";
  }
  return {};
}
} // namespace golly