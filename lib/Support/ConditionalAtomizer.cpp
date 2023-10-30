#include "golly/Support/ConditionalAtomizer.h"

#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>

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
  if (auto as_instr = llvm::dyn_cast<llvm::Instruction>(cond)) {
    auto collection = llvm::SetVector<llvm::Value *>();
    detail::collect(as_instr, collection);
    return collection;
  }
  if (auto as_const = llvm::dyn_cast<llvm::Constant>(cond)) {
    auto type = as_const->getType();
    assert(type->isIntegerTy() && type->getScalarSizeInBits() == 1);
    auto collection = llvm::SetVector<llvm::Value *>();
    collection.insert(cond);
    return collection;
  }

  assert(false);
}
} // namespace golly