#ifndef GOLLY_ANALYSIS_STATEMENTS_H
#define GOLLY_ANALYSIS_STATEMENTS_H
#include <golly/Support/type_list.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/SymbolTableListTraits.h>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>

namespace golly {
using llvm::BasicBlock;
using llvm::Instruction;
using llvm::SymbolTableList;

using std::string;
using std::string_view;
using std::unique_ptr;
using std::variant;

bool is_a_barrier(const llvm::Instruction &instr);

class Statement {
public:
  using InstListType = SymbolTableList<Instruction>;
  using const_iterator = InstListType::const_iterator;

  struct Warp {
    llvm::Value *mask;
  };
  struct Block {};
  struct End {};

  struct Barrier : std::variant<Warp, Block, End> {
    using Base = std::variant<Warp, Block, End>;
    using Base::Base;

    llvm::CallInst *instr;
  };

  enum class Access { Read, Write };
  struct MemoryAccess {
    Access access;
    const llvm::Value *pointer_operand;
  };

  Statement(string_view name, const_iterator begin, const_iterator end);

  string getName() const;
  const_iterator begin() const { return beg_; }
  const_iterator end() const { return end_; }
  const llvm::Instruction &getLastInstruction() const { return *last_; }

  llvm::Optional<Barrier> getBarrier() const { return barrier_; }
  std::span<const MemoryAccess> getAccesses() const { return accesses_; }

private:
  const_iterator beg_, end_, last_;

  // barriers
  llvm::Optional<Barrier> barrier_;

  // memory accesses
  std::vector<MemoryAccess> accesses_;
  const llvm::Value *ptr{};
  const llvm::Value *offset{};

  string name;
};
} // namespace golly
#endif /* GOLLY_ANALYSIS_STATEMENTS_H */
