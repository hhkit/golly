#ifndef ANALYSIS_ERROR_H
#define ANALYSIS_ERROR_H
#include <llvm/IR/Instruction.h>
#include <vector>

namespace golly {
struct Error {
  enum class Type {
    BarrierDivergence,
    DataRace,
  };

  Type type;
  llvm::Instruction *instr{};
};

using ErrorList = std::vector<Error>;
} // namespace golly

#endif /* ANALYSIS_ERROR_H */
