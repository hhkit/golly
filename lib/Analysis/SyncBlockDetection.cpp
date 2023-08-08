#include <fmt/format.h>
#include <golly/Analysis/SyncBlockDetection.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>
#define DEBUG_TYPE "golly-detection"

namespace golly {
namespace detail {

static unique_ptr<Statement> build(const BasicBlock &bb) {
  auto beg = bb.begin();
  auto itr = beg;
  auto first = unique_ptr<Statement>();

  int count{};
  Statement *prev{};

  auto append = [&](unsigned type, BasicBlock::const_iterator rbegin,
                    BasicBlock::const_iterator rend) {
    const auto name = fmt::format("{}_{}", bb.getName().str(), count++);
    auto newNode = Statement::create(type, &bb, rbegin, rend, name);

    if (prev) {
      prev->addSuccessor(std::move(newNode));
    }
    prev = newNode.get();

    if (!first)
      first = std::move(newNode);
  };

  while (itr != bb.end()) {
    if (Statement::isStatementDivider(*itr)) {
      auto type_index = dividerIndex(*itr);
      llvm::dbgs() << *itr << " -- has type " << type_index << "\n";
      append(type_index, beg, (++itr));
      beg = itr;
      continue;
    }
    ++itr;
  }

  if (beg != bb.end())
    append(StatementTypes::length, beg, bb.end());

  return first;
}
} // namespace detail

AnalysisKey StatementDetectionPass::Key;
void StatementDetection::analyze(const Function &f) {
  for (auto &bb : f) {
    map[&bb] = detail::build(bb);
  }
}

Statement *StatementDetection::getTopLevelPhase(const BasicBlock &bb) const {
  if (auto itr = map.find(&bb); itr != map.end()) {
    return itr->second.get();
  }
  return nullptr;
}
StatementDetection::StatementRange
StatementDetection::iterateStatements(const BasicBlock &f) const {
  if (auto itr = map.find(&f); itr != map.end()) {
    return StatementRange{itr->second.get()};
  }
  return StatementRange{};
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

StatementDetection::StatementRange::iterator
StatementDetection::StatementRange::begin() const {
  return iterator{ptr_};
}

StatementDetection::StatementRange::iterator
StatementDetection::StatementRange::end() const {
  return iterator{};
}

StatementDetectionPass::Result
StatementDetectionPass::run(Function &f, FunctionAnalysisManager &fam) {
  StatementDetection sbd;
  sbd.analyze(f);
  return sbd;
}
} // namespace golly