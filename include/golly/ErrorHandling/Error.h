#ifndef GOLLY_ERRORHANDLING_ERROR_H
#define GOLLY_ERRORHANDLING_ERROR_H
#include "golly/Support/isl.h"
#include <llvm/Support/raw_ostream.h>
#include <variant>
#include <vector>

namespace golly {

class Statement;

enum class Level {
  Grid,
  Block,
  Warp,
};

struct UnreachableBarrier {
  const Statement *barrier;
  llvm::raw_ostream &print(llvm::raw_ostream &) const;
};

struct BarrierDivergence {
  const Statement *barrier;
  Level level;

  llvm::raw_ostream &print(llvm::raw_ostream &) const;
};

struct DataRace {
  const Statement *instr1;
  const Statement *instr2;
  Level level;
  islpp::basic_map clashing_tids;

  llvm::raw_ostream &print(llvm::raw_ostream &) const;
};

struct Error : std::variant<BarrierDivergence, DataRace, UnreachableBarrier> {
  using Base = std::variant<BarrierDivergence, DataRace, UnreachableBarrier>;
  using Base::Base;

  llvm::raw_ostream &print(llvm::raw_ostream &os) const {
    return std::visit(
        [&](auto &e) -> llvm::raw_ostream & { return e.print(os); }, *this);
  }
};

struct ErrorList : std::vector<Error> {
  using Base = std::vector<Error>;
  using Base::Base;

  explicit operator bool() const {
    for (auto &elem : *this) {
      if (elem.index() != 2)
        return true;
    }
    return false;
  }
};
} // namespace golly

#endif /* GOLLY_ERRORHANDLING_ERROR_H */
