#include <array>
#include <golly/Analysis/Statements.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/IR/Instructions.h>
namespace golly {

namespace detail {
using InstListType = SymbolTableList<Instruction>;
using const_iterator = InstListType::const_iterator;

template <typename... Ts> auto make_create_statement_lut(type_list<Ts...>) {
  using CreateTy = unique_ptr<Statement>(const BasicBlock *bb, const_iterator b,
                                         const_iterator e, string_view name);
  return std::array<CreateTy *, sizeof...(Ts) + 1>{
      (+[](const BasicBlock *bb, const_iterator b, const_iterator e,
           string_view name) {
        return unique_ptr<Statement>(new Ts(bb, b, e, name));
      })...,
      (+[](const BasicBlock *bb, const_iterator b, const_iterator e,
           string_view name) {
        return unique_ptr<Statement>(new Statement(bb, b, e, name));
      })};
}
} // namespace detail

bool Statement::isStatementDivider(const llvm::Instruction &instr) {
  return detail::dividerIndex(instr) != StatementTypes::length;
}

unique_ptr<Statement> Statement::create(unsigned index, const BasicBlock *bb,
                                        const_iterator b, const_iterator e,
                                        string_view name) {
  static auto lut = detail::make_create_statement_lut(StatementTypes{});

  return lut[index](bb, b, e, name);
}

bool BarrierStatement::isDivider(const llvm::Instruction &instr) {
  static llvm::StringSet barrier_instructions{"llvm.nvvm.bar.warp.sync",
                                              "_Z10__syncwarpj"};

  auto as_fn = llvm::dyn_cast_or_null<llvm::CallInst>(&instr);

  if (!as_fn)
    return llvm::isa<llvm::ReturnInst>(instr);

  if (const auto called_fn = as_fn->getCalledFunction()) {
    return barrier_instructions.contains(called_fn->getName());
  }
  return false;
}

bool StoreStatement::isDivider(const llvm::Instruction &instr) {
  return llvm::isa<llvm::StoreInst>(instr);
}

bool LoadStatement::isDivider(const llvm::Instruction &instr) {
  return llvm::isa<llvm::LoadInst>(instr);
}

Statement::Statement(const BasicBlock *bb, const_iterator b, const_iterator e,
                     string_view name)
    : bb_{bb}, beg_{b}, end_{e}, name{name} {
  assert((beg_ != end_) && "No trivial statements");
  for (last_ = b++; b != e; ++last_, ++b)
    ;
}

void Statement::addSuccessor(unique_ptr<Statement> child) {
  successor = std::move(child);
}

string Statement::getName() const { return string{getSuffix()} + name; }

Statement *Statement::getSuccessor() const { return successor.get(); }

} // namespace golly