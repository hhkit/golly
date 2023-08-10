#include <array>
#include <golly/Analysis/Statements.h>
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

template <typename... Ts>
constexpr auto make_create_statement_lut(type_list<Ts...>) {
  using CreateTy = unique_ptr<Statement>(const StatementConfig &);
  return std::array<CreateTy *, sizeof...(Ts) + 1>{
      ([](const StatementConfig &cfg) {
        return unique_ptr<Statement>(new Ts(cfg));
      })...,
      ([](const StatementConfig &cfg) {
        return unique_ptr<Statement>(new Statement(cfg));
      })};
}

BarrierStatement::Barrier
createBarrierMetadata(const llvm::Instruction &barrier_instr) {
  const auto as_fn = llvm::dyn_cast_or_null<llvm::CallInst>(&barrier_instr);

  if (!as_fn) {
    const auto as_ret =
        llvm::dyn_cast_or_null<llvm::ReturnInst>(&barrier_instr);
    if (as_ret)
      return BarrierStatement::End{};
  }

  assert(as_fn && "defining instruction should be at least an intrinsic call");

  const auto fn_name = as_fn->getCalledFunction()->getName();

  if (warp_instructions.contains(fn_name)) {
    assert(as_fn->getCalledFunction()->arg_size() == 1 &&
           "warp barrier should always have a mask");
    auto mask = as_fn->getArgOperand(0);

    // retrieve mask
    return BarrierStatement::Warp{};
  }

  if (block_instructions.contains(fn_name)) {
    return BarrierStatement::Block();
  }

  return BarrierStatement::Block{};
}
} // namespace detail

bool Statement::isStatementDivider(const llvm::Instruction &instr) {
  return detail::dividerIndex(instr) != StatementTypes::length;
}

unique_ptr<Statement> Statement::create(unsigned index,
                                        const StatementConfig &cfg) {
  constexpr auto lut = detail::make_create_statement_lut(StatementTypes{});

  return lut[index](cfg);
}

BarrierStatement::BarrierStatement(const StatementConfig &cfg)
    : Base{cfg}, barrier_{
                     detail::createBarrierMetadata(getDefiningInstruction())} {}

bool BarrierStatement::isDivider(const llvm::Instruction &instr) {

  auto as_fn = llvm::dyn_cast_or_null<llvm::CallInst>(&instr);

  if (!as_fn)
    return llvm::isa<llvm::ReturnInst>(instr);

  if (const auto called_fn = as_fn->getCalledFunction()) {
    return detail::warp_instructions.contains(called_fn->getName()) ||
           detail::block_instructions.contains(called_fn->getName());
  }
  return false;
}

bool MemoryAccessStatement::isDivider(const llvm::Instruction &instr) {
  return llvm::isa<llvm::StoreInst>(instr) || llvm::isa<llvm::LoadInst>(instr);
}

string_view MemoryAccessStatement::getSuffix() const {
  switch (getAccessType()) {
  case Access::Read:
    return "LOAD.";
  case Access::Write:
    return "STORE.";
  }
  assert(false && "memory access should only be load or store");
  return {};
}

MemoryAccessStatement::Access MemoryAccessStatement::getAccessType() const {
  if (auto as_store = llvm::dyn_cast_or_null<llvm::StoreInst>(
          &this->getDefiningInstruction()))
    return Access::Write;
  if (auto as_load = llvm::dyn_cast_or_null<llvm::LoadInst>(
          &this->getDefiningInstruction()))
    return Access::Read;

  assert(false && "memory access should only be load or store");
  return {};
}

const llvm::Value *MemoryAccessStatement::getPointerOperand() const {
  if (auto as_store = llvm::dyn_cast_or_null<llvm::StoreInst>(
          &this->getDefiningInstruction()))
    return as_store->getPointerOperand();
  if (auto as_load = llvm::dyn_cast_or_null<llvm::LoadInst>(
          &this->getDefiningInstruction()))
    return as_load->getPointerOperand();

  assert(false && "memory access should only be load or store");
  return nullptr;
}

Statement::Statement(const StatementConfig &cfg)
    : bb_{cfg.bb}, beg_{cfg.begin}, end_{cfg.end}, name{cfg.name} {
  assert((beg_ != end_) && "No trivial statements");
  auto b = cfg.begin;
  auto e = cfg.end;
  for (last_ = b++; b != e; ++last_, ++b)
    ;
}

void Statement::addSuccessor(unique_ptr<Statement> child) {
  successor = std::move(child);
}

string Statement::getName() const { return string{getSuffix()} + name; }

Statement *Statement::getSuccessor() const { return successor.get(); }

} // namespace golly