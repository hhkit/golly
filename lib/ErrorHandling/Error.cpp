#include "golly/ErrorHandling/Error.h"
#include <llvm/Support/WithColor.h>

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

llvm::raw_ostream &UnreachableBarrier::print(llvm::raw_ostream &os) const {
  return os << "unreachable barrier";
}

llvm::raw_ostream &BarrierDivergence::print(llvm::raw_ostream &os) const {
  return os << print_level(level) << " barrier divergence";
}

llvm::raw_ostream &DataRace::print(llvm::raw_ostream &os) const {
  return os << print_level(level) << " data race";
}
} // namespace golly