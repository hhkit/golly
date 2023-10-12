#include "golly/Support/ConditionalAtomizer.h"

#include <llvm/IR/Instructions.h>

namespace golly {
namespace detail {
void collect(llvm::Instruction *instr, llvm::SetVector<llvm::Value *> &set) {
  switch (instr->getOpcode()) {
  case llvm::BinaryOperator::And:
  case llvm::BinaryOperator::Or:
    collect(llvm::dyn_cast<llvm::Instruction>(instr->getOperand(0)), set);
    collect(llvm::dyn_cast<llvm::Instruction>(instr->getOperand(1)), set);
    return;
  default:
    set.insert(instr);
    return;
  }
}
} // namespace detail

llvm::SetVector<llvm::Value *> atomize(llvm::Value *cond) {
  auto as_instr = llvm::dyn_cast<llvm::Instruction>(cond);
  assert(as_instr);
  auto collection = llvm::SetVector<llvm::Value *>();
  detail::collect(as_instr, collection);
  return collection;
}
} // namespace golly