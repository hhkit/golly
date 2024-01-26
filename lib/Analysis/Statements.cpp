
#include "golly/Analysis/Statements.h"
#include <array>
#include <llvm-14/llvm/IR/Instruction.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Debug.h>
namespace golly {

namespace detail {
using InstListType = SymbolTableList<Instruction>;
using const_iterator = InstListType::const_iterator;
static llvm::StringSet warp_instructions{
    "llvm.nvvm.bar.warp.sync",
    "_Z10__syncwarpj" // TODO: better way of detecting this - maybe
                      // canonicalize?
};
static llvm::StringSet block_instructions{
    "llvm.nvvm.barrier0",
};

Statement::Barrier
createBarrierMetadata(const llvm::Instruction &barrier_instr) {
  const auto as_fn = llvm::dyn_cast_or_null<llvm::CallInst>(&barrier_instr);

  if (!as_fn && llvm::isa<llvm::ReturnInst>(&barrier_instr))
    return Statement::End{};

  assert(as_fn && "defining instruction should be at least an intrinsic call");

  const auto fn_name = as_fn->getCalledFunction()->getName();

  if (warp_instructions.contains(fn_name)) {
    assert(as_fn->getCalledFunction()->arg_size() == 1 &&
           "warp barrier should always have a mask");

    // retrieve mask
    return Statement::Warp{as_fn->getArgOperand(0)};
  }

  if (block_instructions.contains(fn_name))
    return Statement::Block();

  assert(false);
  return {};
}

Statement::MemoryAccess createAccessMetadata(const llvm::Instruction &inst) {
  assert(llvm::isa<llvm::LoadInst>(inst) || llvm::isa<llvm::StoreInst>(inst));
  const bool is_a_load = llvm::isa<llvm::LoadInst>(inst);

  Statement::Access access =
      is_a_load ? Statement::Access::Read : Statement::Access::Write;

  const llvm::Value *ptr_value =
      is_a_load ? llvm::cast<llvm::LoadInst>(inst).getPointerOperand()
                : llvm::cast<llvm::StoreInst>(inst).getPointerOperand();

  return Statement::MemoryAccess{
      .access = access,
      .pointer_operand = ptr_value,
  };
}

Statement::MemoryAccess createAccessMetadata(const llvm::AtomicRMWInst &inst) {
  const llvm::Value *ptr_value = inst.getPointerOperand();

  return Statement::MemoryAccess{
      .access = Statement::Access::Atomic,
      .pointer_operand = ptr_value,
  };
}
} // namespace detail

bool is_a_barrier(const llvm::Instruction &inst) {
  if (auto as_call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
    auto name = as_call->getCalledFunction()->getName();
    return detail::block_instructions.contains(name) ||
           detail::warp_instructions.contains(name);
  }
  return false;
}

Statement::Statement(string_view name, const_iterator begin, const_iterator end)
    : beg_{begin}, end_{end}, name{name} {
  assert((beg_ != end_) && "No trivial statements");
  auto b = begin;
  auto e = end;

  for (last_ = b++; b != end; ++b, ++last_)
    ;

  for (auto i = begin; i != end; ++i) {
    switch (i->getOpcode()) {
    case llvm::Instruction::Load:
    case llvm::Instruction::Store:
      accesses_.emplace_back(detail::createAccessMetadata(*i));
      continue;
    case llvm::Instruction::AtomicRMW:
      accesses_.emplace_back(
          detail::createAccessMetadata(llvm::cast<llvm::AtomicRMWInst>(*i)));
      continue;
    case llvm::Instruction::Call:
      if (is_a_barrier(*i)) {
        barrier_ = detail::createBarrierMetadata(*i);
        barrier_->instr = &llvm::cast<llvm::CallInst>(*i);
      }
      continue;
    case llvm::Instruction::Ret:
      barrier_ = detail::createBarrierMetadata(*i);
      barrier_->instr = &llvm::cast<llvm::ReturnInst>(*i);
      continue;
    default:
      break;
    }
  }
}

string Statement::getName() const { return name; }

} // namespace golly