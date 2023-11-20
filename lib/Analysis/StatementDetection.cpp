#include <fmt/format.h>
#include <golly/Analysis/StatementDetection.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#define DEBUG_TYPE "golly-detection"

namespace golly {
namespace detail {

using divide_fn = bool(const llvm::Instruction &instr);

constexpr auto divider_lut = std::array{
    // low speed
    +[](const llvm::Instruction &instr) -> bool {
      return is_a_barrier(instr) || llvm::isa<llvm::StoreInst>(instr) ||
             llvm::isa<llvm::LoadInst>(instr);
    },
    +[](const llvm::Instruction &instr) -> bool {
      return is_a_barrier(instr) || llvm::isa<llvm::StoreInst>(instr);
    },
    +[](const llvm::Instruction &instr) -> bool { return is_a_barrier(instr); },
};

static int speed = 1;

static std::vector<Statement> build(const BasicBlock &bb) {
  std::vector<Statement> statements;

  auto front = bb.begin();
  auto itr = front;

  auto divide = divider_lut[speed];

  while (itr != bb.end()) {
    if (divide(*itr)) {
      statements.emplace_back(
          llvm::formatv("{0}_{1}", bb.getName(), statements.size()).str(),
          front, ++itr);
      front = itr;
      continue;
    }
    ++itr;
  }

  if (front != bb.end())
    statements.emplace_back(
        llvm::formatv("{0}_{1}", bb.getName(), statements.size()).str(), front,
        bb.end());

  return statements;
}
} // namespace detail

AnalysisKey StatementDetectionPass::Key;
void StatementDetection::analyze(const Function &f) {
  for (auto &bb : f) {
    map[&bb] = detail::build(bb);
  }

  for (auto &[bb, _] : map) {
    for (auto &stmt : iterateStatements(*bb))
      cached_names[stmt.getName()] = &stmt;
  }
}

const Statement *StatementDetection::getStatement(string_view name) const {
  if (auto itr = cached_names.find(name); itr != cached_names.end()) {
    return itr->second;
  }
  return nullptr;
}

std::span<const Statement>
StatementDetection::iterateStatements(const BasicBlock &f) const {
  if (auto itr = map.find(&f); itr != map.end())
    return itr->second;

  return {};
}

llvm::raw_ostream &StatementDetection::dump(llvm::raw_ostream &os) const {
  for (auto &&[bb, sbs] : map) {
    os << bb << "\n";

    for (auto &sb : iterateStatements(*bb)) {
      os << sb.getName() << ":\n";
      for (auto &instr : sb) {
        os << "\t" << instr << "\n";
      }
      os << "----end sb\n ";
    }
  }
  return os;
}

StatementDetectionPass::Result
StatementDetectionPass::run(Function &f, FunctionAnalysisManager &fam) {
  StatementDetection sbd;
  sbd.analyze(f);
  // sbd.dump(llvm::dbgs());
  return sbd;
}
} // namespace golly