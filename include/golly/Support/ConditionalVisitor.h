#ifndef GOLLY_SUPPORT_CONDITIONALVISITOR_H
#define GOLLY_SUPPORT_CONDITIONALVISITOR_H

#include <llvm/IR/InstVisitor.h>

namespace golly {
template <typename T, typename Ret>
class ConditionalVisitor : public llvm::InstVisitor<T, Ret> {
public:
  using Base = llvm::InstVisitor<T, Ret>;
  using Base::Base;
  using Base::visit;

  virtual Ret visitAnd(llvm::Instruction &) = 0;
  virtual Ret visitOr(llvm::Instruction &) = 0;
  virtual Ret visitICmpInst(llvm::ICmpInst &) = 0;
  // virtual Ret visitXor(llvm::Instruction *) = 0; // ?
  Ret visitBinaryOperator(llvm::BinaryOperator &inst) {
    switch (inst.getOpcode()) {
    case llvm::Instruction::BinaryOps::And:
      return visitAnd(inst);
    case llvm::Instruction::BinaryOps::Or:
      return visitOr(inst);
    default:
      return Base::visitBinaryOperator(inst);
    }
  }
};
} // namespace golly

#endif /* GOLLY_SUPPORT_CONDITIONALVISITOR_H */
