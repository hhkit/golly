#include <fmt/format.h>
#include <golly/Analysis/SyncBlockDetection.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>
#define DEBUG_TYPE "golly-detection"

namespace golly {

namespace detail {
using llvm::StringSet;
static bool isBarrier(const llvm::Instruction &instr) {
  static StringSet barrier_instructions{"llvm.nvvm.bar.warp.sync",
                                        "_Z10__syncwarpj"};

  auto as_fn = llvm::dyn_cast_or_null<llvm::CallInst>(&instr);

  if (!as_fn)
    return llvm::isa<llvm::ReturnInst>(instr);

  if (const auto called_fn = as_fn->getCalledFunction()) {
    // llvm::dbgs() << "called_fn " << called_fn->getName() << "\n";
    return barrier_instructions.contains(called_fn->getName());
  }
  return false;
}

static unique_ptr<SyncBlock> build(const BasicBlock &bb) {
  auto beg = bb.begin();
  auto itr = beg;
  auto first = unique_ptr<SyncBlock>();

  int count{};
  SyncBlock *prev{};

  auto append = [&](BasicBlock::const_iterator rbegin,
                    BasicBlock::const_iterator rend) {
    auto newNode = std::make_unique<SyncBlock>(
        rbegin, rend, fmt::format("{}_{}", bb.getName().str(), count++));

    if (prev) {
      prev->addSuccessor(std::move(newNode));
    }
    prev = newNode.get();

    if (!first)
      first = std::move(newNode);
  };

  while (itr != bb.end()) {
    if (isBarrier(*itr)) {
      append(beg, (++itr));
      beg = itr;
      continue;
    }
    ++itr;
  }

  if (beg != bb.end())
    append(beg, bb.end());

  return first;
}
} // namespace detail

SyncBlock::SyncBlock(const_iterator b, const_iterator e, string_view name)
    : beg_{b}, end_{e}, name{name} {
  assert((beg_ != end_) && "No trivial sync blocks");
  for (last_ = b++; b != e; ++last_, ++b)
    ;
}

void SyncBlock::addSuccessor(unique_ptr<SyncBlock> child) {
  successor = std::move(child);
}

string_view SyncBlock::getName() const { return name; }

SyncBlock *SyncBlock::getSuccessor() const { return successor.get(); }

bool SyncBlock::willSynchronize() const {
  // llvm::dbgs() << "SYNCTEST" << *last_ << "\n";
  return detail::isBarrier(*last_);
}

AnalysisKey SyncBlockDetectionPass::Key;
void SyncBlockDetection::analyze(const Function &f) {
  for (auto &bb : f) {
    map[&bb] = detail::build(bb);
  }
}

SyncBlock *SyncBlockDetection::getTopLevelPhase(const BasicBlock &bb) const {
  if (auto itr = map.find(&bb); itr != map.end()) {
    return itr->second.get();
  }
  return nullptr;
}
SyncBlockDetection::SyncBlockRange
SyncBlockDetection::iterateSyncBlocks(const BasicBlock &f) const {
  if (auto itr = map.find(&f); itr != map.end()) {
    return SyncBlockRange{itr->second.get()};
  }
  return SyncBlockRange{};
}

llvm::raw_ostream &SyncBlockDetection::dump(llvm::raw_ostream &os) const {
  for (auto &&[bb, sbs] : map) {
    os << bb << "\n";

    for (auto &sb : iterateSyncBlocks(*bb)) {
      os << sb.getName() << ":\n";
      for (auto &instr : sb) {
        os << "\t" << instr << "\n";
      }
      if (sb.willSynchronize())
        os << "----synchronizes\n";
      os << "----end sb\n ";
    }
  }
  return os;
}

SyncBlockDetection::SyncBlockRange::iterator
SyncBlockDetection::SyncBlockRange::begin() const {
  return iterator{ptr_};
}

SyncBlockDetection::SyncBlockRange::iterator
SyncBlockDetection::SyncBlockRange::end() const {
  return iterator{};
}

SyncBlockDetectionPass::Result
SyncBlockDetectionPass::run(Function &f, FunctionAnalysisManager &fam) {
  SyncBlockDetection sbd;
  sbd.analyze(f);
  return sbd;
}
} // namespace golly