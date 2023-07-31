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

  return as_fn ? barrier_instructions.contains(
                     as_fn->getCalledFunction()->getName())
               : llvm::isa<llvm::ReturnInst>(instr);
}

static unique_ptr<SyncBlock> build(const BasicBlock &bb) {
  auto beg = bb.begin();
  auto itr = beg;
  auto first = unique_ptr<SyncBlock>();
  SyncBlock *prev{};

  auto append = [&](BasicBlock::const_iterator rbegin,
                    BasicBlock::const_iterator rend) {
    auto newNode = std::make_unique<SyncBlock>(rbegin, rend);

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

SyncBlock::SyncBlock(const_iterator b, const_iterator e) : beg_{b}, end_{e} {
  assert((beg_ != end_) && "No trivial sync blocks");
  for (last_ = ++b; b != e; ++last_, ++b)
    ;
}

void SyncBlock::addSuccessor(unique_ptr<SyncBlock> child) {
  successor = std::move(child);
}

SyncBlock *SyncBlock::getSuccessor() const { return successor.get(); }

bool SyncBlock::willSynchronize() const { return detail::isBarrier(*last_); }

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

llvm::raw_ostream &SyncBlockDetection::dump(llvm::raw_ostream &os) const {
  for (auto &&[bb, sbs] : map) {
    auto itr = sbs.get();

    os << bb << "\n";
    while (itr) {
      for (auto &instr : *itr) {
        os << "\tsb" << instr << "\n";
      }
      os << "----end sb\n ";
      itr = itr->getSuccessor();
    }
  }
  return os;
}

SyncBlockDetectionPass::Result
SyncBlockDetectionPass::run(Function &f, FunctionAnalysisManager &fam) {
  SyncBlockDetection sbd;
  sbd.analyze(f);
  return sbd;
}
} // namespace golly