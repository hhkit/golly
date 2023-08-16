#ifndef GOLLY_ANALYSIS_STATEMENTS_H
#define GOLLY_ANALYSIS_STATEMENTS_H
#include <golly/Support/type_list.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/SymbolTableListTraits.h>
#include <memory>
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

using StatementTypes =
    detail::type_list<class BarrierStatement, class MemoryAccessStatement>;

struct StatementConfig {
  using InstListType = SymbolTableList<Instruction>;
  using const_iterator = InstListType::const_iterator;

  const BasicBlock *bb;
  const_iterator begin, end;
  string_view name;
};

class Statement {
public:
  using InstListType = SymbolTableList<Instruction>;
  using const_iterator = InstListType::const_iterator;

  Statement(const StatementConfig &);
  virtual ~Statement() = default;

  void addSuccessor(unique_ptr<Statement> child);
  Statement *getSuccessor() const;

  virtual size_t getTypeIndex() const { return StatementTypes::length; };
  template <typename T> bool isA() const {
    return getTypeIndex() == StatementTypes::index_of<T>();
  }
  template <typename T> T *as() {
    return static_cast<T *>(isA<T>() ? this : nullptr);
  }

  template <typename T> const T *as() const {
    return static_cast<const T *>(isA<T>() ? this : nullptr);
  }

  string getName() const;
  const_iterator begin() const { return beg_; }
  const_iterator end() const { return end_; }
  const BasicBlock *getBlock() const { return bb_; }

  static bool isStatementDivider(const Instruction &instr);
  static unique_ptr<Statement> create(unsigned index, const StatementConfig &);
  const Instruction &getDefiningInstruction() const { return *last_; }

protected:
  virtual string_view getSuffix() const { return ""; }

private:
  const_iterator beg_, end_, last_;
  const BasicBlock *bb_;
  unique_ptr<Statement> successor;
  string name;
};

// crtp to enable type indexing
template <typename T> class BaseStatement : public Statement {
public:
  using Statement::Statement;
  using Base = BaseStatement;

  size_t getTypeIndex() const override { return StatementTypes::index_of<T>(); }
};

// this statement contains a barrier
class BarrierStatement : public BaseStatement<BarrierStatement> {
public:
  struct Warp {
    llvm::Value *mask;
  };
  struct Block {};
  struct End {};

  using Barrier = variant<Warp, Block, End>;

  BarrierStatement(const StatementConfig &rhs);
  static bool isDivider(const llvm::Instruction &);
  string_view getSuffix() const { return "Sync."; }
  const Barrier &getBarrier() const { return barrier_; }

private:
  Barrier barrier_;
};

// this statement contains a memory access
class MemoryAccessStatement : public BaseStatement<MemoryAccessStatement> {
public:
  enum class Access {
    Read,
    Write,
  };

  using Base::Base;
  static bool isDivider(const llvm::Instruction &);
  string_view getSuffix() const;
  const llvm::Value *getPointerOperand() const;

  Access getAccessType() const;
};

namespace detail {
template <typename... Ts>
inline auto dividerIndexImpl(const llvm::Instruction &instr, type_list<Ts...>) {
  using TL = type_list<Ts...>;
  auto isDivider = (Ts::isDivider(instr) || ...);
  if (isDivider)
    return ((Ts::isDivider(instr) ? TL::template index_of<Ts>() : 0) + ...);
  else
    return sizeof...(Ts);
}

inline auto dividerIndex(const llvm::Instruction &instr) {
  return dividerIndexImpl(instr, StatementTypes{});
}
} // namespace detail
} // namespace golly
#endif /* GOLLY_ANALYSIS_STATEMENTS_H */
