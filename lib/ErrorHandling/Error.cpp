#include "golly/ErrorHandling/Error.h"
#include "golly/Analysis/Statements.h"
#include <llvm-14/llvm/Support/WithColor.h>
#include <llvm/DebugInfo/DIContext.h>
#include <llvm/IR/DebugInfoMetadata.h>

namespace golly {

static std::string_view print_level(Level level) {
  switch (level) {
  case Level::Grid:
    return "Global";
  case Level::Block:
    return "Block-level";
  case Level::Warp:
    return "Warp-level";
  }
  return "";
}
static llvm::Optional<std::string> print_location(const llvm::DebugLoc &dl) {
  if (auto diloc = dl.get()) {
    std::string loc = diloc->getFilename().str();
    loc += ":" + std::to_string(diloc->getLine());
    loc += ":" + std::to_string(diloc->getColumn());
    return loc;
  }
  return llvm::None;
}

llvm::raw_ostream &UnreachableBarrier::print(llvm::raw_ostream &os) const {
  return os << "unreachable barrier";
}

llvm::raw_ostream &BarrierDivergence::print(llvm::raw_ostream &os) const {

  llvm::WithColor(os, llvm::HighlightColor::Warning)
      << print_level(level) << " barrier divergence";
  os << " observed in \n";
  os << "\t" << print_location(barrier->getLastInstruction().getDebugLoc())
     << "\n";
  return os;
}

llvm::raw_ostream &DataRace::print(llvm::raw_ostream &os) const {
  llvm::WithColor(os, llvm::HighlightColor::Warning)
      << print_level(level) << " data race";
  os << " detected between:\n";
  os << "\t" << print_location(instr1->getLastInstruction().getDebugLoc())
     << " and \n";
  os << "\t" << print_location(instr2->getLastInstruction().getDebugLoc())
     << "\n";
  return os;
}
} // namespace golly