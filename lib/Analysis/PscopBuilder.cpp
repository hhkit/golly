#include <fmt/format.h>
#include <golly/Analysis/PscopBuilder.h>
#include <golly/Analysis/SyncBlockDetection.h>
#include <golly/Support/isl_llvm.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/PriorityQueue.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/RegionInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/IR/Instructions.h>
#include <queue>
#include <stack>
#include <string>
#include <string_view>
#include <vector>

namespace golly {
using llvm::LoopInfo;
using llvm::Optional;
using llvm::RegionInfo;
using llvm::ScalarEvolution;
using std::stack;
using std::string;
using std::string_view;
using std::vector;

using InductionVarSet = llvm::MapVector<const llvm::Value *, string>;

namespace detail {

class IVarDatabase {
public:
  bool addThreadIdentifier(string_view intrinsic, string_view name,
                           int max = 255) {
    if (thread_ids.try_emplace(intrinsic, string(name)).second) {
      domain = flat_cross(
          domain,
          islpp::set{fmt::format("{{ [{0}] : 0 <= {0} <= {1} }}", name, max)});
      return true;
    }
    return false;
  }

  void addInductionVariable(const llvm::Value *val) {
    if (induction_vars.find(val) == induction_vars.end()) {
      const auto &name = induction_vars[val] = fmt::format("i{}", id++);
      domain = add_dims(domain, islpp::dim::set, {name});
    }
  }

  void addConstraint(islpp::set constraining_set) {
    domain = domain * constraining_set;
  }

  bool isIVar(const llvm::Value *val) const {
    if (induction_vars.find(val) != induction_vars.end())
      return true;

    if (const auto callInst = llvm::dyn_cast<llvm::CallInst>(val)) {
      return thread_ids.find(callInst->getCalledFunction()->getName()) !=
             thread_ids.end();
    }

    return false;
  }

  string_view getName(const llvm::Value *val) const {
    if (auto itr = induction_vars.find(val); itr != induction_vars.end())
      return itr->second;

    if (const auto callInst = llvm::dyn_cast<llvm::CallInst>(val)) {
      auto itr = thread_ids.find(callInst->getCalledFunction()->getName());
      if (itr != thread_ids.end())
        return itr->second;
    }

    return "";
  }

  string getVector() const {
    vector<string> tmp;
    tmp.reserve(thread_ids.size() + induction_vars.size());
    for (auto &entry : thread_ids)
      tmp.emplace_back(entry.second);

    for (auto &&[k, v] : induction_vars)
      tmp.emplace_back(v);

    return fmt::format("[{}]", fmt::join(tmp, ","));
  }

  const islpp::set &getDomain() const { return domain; }

private:
  llvm::StringMap<string> thread_ids;
  InductionVarSet induction_vars;
  islpp::set domain{"{ [] }"};

  int id = 0;
};

} // namespace detail

enum class QuaffClass {
  CInt,  // compile-time constant integer
  IVar,  // Instantiation variable, we distinguish instantiation from induction
         // because they also include thread identifiers
  Param, // run-time constant parameter
  Invalid, // not quasi-affine
};

class QuaffExpr {
public:
  QuaffClass type;
  islpp::pw_aff expr{"{ [] -> [0] }"};
  vector<string> params;

  bool isValid() const { return type != QuaffClass::Invalid; }
  explicit operator bool() const { return isValid(); }
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, const QuaffExpr &e) {
  return o << e.expr;
}

struct RegionInvariants {
  Region *region;
  islpp::union_set iteration_domain;
  detail::IVarDatabase induction_variables;
  int depth = 0;

  RegionInvariants(Region *reg = nullptr) : region{reg} {}
  RegionInvariants clone(Region *child) const {
    auto ret = *this;
    ret.region = child;
    ret.depth += 1;
    return ret;
  }
};

class QuaffBuilder : public llvm::SCEVVisitor<QuaffBuilder, QuaffExpr> {
public:
  QuaffBuilder(RegionInvariants &ri, detail::IVarDatabase &ivdb,
               llvm::ScalarEvolution &se)
      : ri{ri}, ivdb{ivdb}, se{se} {
    indvars = ivdb.getVector();
  }

  QuaffExpr visitConstant(const llvm::SCEVConstant *cint) {
    auto expr = fmt::format("{{ {} -> [{}]  }}", indvars,
                            cint->getAPInt().getSExtValue());
    return QuaffExpr{QuaffClass::CInt, islpp::pw_aff{expr}};
  }

  QuaffExpr visitAddExpr(const llvm::SCEVAddExpr *S) {
    return mergeNary(S, [](auto lhs, auto rhs) { return lhs + rhs; });
  }

  QuaffExpr visitMulExpr(const llvm::SCEVMulExpr *S) {
    auto lhs = visit(S->getOperand(0));
    auto rhs = visit(S->getOperand(1));

    // check for invalid
    auto newClass = std::max(lhs.type, rhs.type);

    if (newClass == QuaffClass::Invalid)
      return QuaffExpr{QuaffClass::Invalid};

    if (lhs.type == QuaffClass::Param) {
      if (rhs.type == QuaffClass::IVar)
        return QuaffExpr{QuaffClass::Invalid};

      // else expr is param * cint or param * param which is param
      return QuaffExpr{QuaffClass::Param, lhs.expr * rhs.expr};
    }

    if (lhs.type == QuaffClass::IVar) {
      if (rhs.type == QuaffClass::CInt)
        return QuaffExpr{QuaffClass::IVar, lhs.expr * rhs.expr};

      // else expr is ivar * param or ivar * ivar which is invalid
      return QuaffExpr{QuaffClass::Invalid};
    }

    // lhs must be CInt
    // always valid
    return QuaffExpr(rhs.type, lhs.expr * rhs.expr);
  }

  QuaffExpr visitPtrToIntExpr(const llvm::SCEVPtrToIntExpr *S) {
    return visit(S->getOperand());
  }
  QuaffExpr visitTruncateExpr(const llvm::SCEVTruncateExpr *S) {
    return visit(S->getOperand());
  }
  QuaffExpr visitZeroExtendExpr(const llvm::SCEVZeroExtendExpr *S) {
    return visit(S->getOperand());
  }
  QuaffExpr visitSignExtendExpr(const llvm::SCEVSignExtendExpr *S) {
    return visit(S->getOperand());
  }
  QuaffExpr visitSMaxExpr(const llvm::SCEVSMaxExpr *S) {
    return mergeNary(S, [](auto lhs, auto rhs) { return max(lhs, rhs); });
  }
  QuaffExpr visitUMaxExpr(const llvm::SCEVUMaxExpr *S) {
    return mergeNary(S, [](auto lhs, auto rhs) { return max(lhs, rhs); });
  }
  QuaffExpr visitSMinExpr(const llvm::SCEVSMinExpr *S) {
    return mergeNary(S, [](auto lhs, auto rhs) { return max(lhs, rhs); });
  }
  QuaffExpr visitUMinExpr(const llvm::SCEVUMinExpr *S) {
    return mergeNary(S, [](auto lhs, auto rhs) { return max(lhs, rhs); });
  }

  QuaffExpr visitUDivExpr(const llvm::SCEVUDivExpr *S) {
    // unsigned division
    const auto dividend = S->getLHS();
    const auto divisor = S->getRHS();

    // divisor must be const to be affine
    return visitDivision(dividend, divisor, S);
  }
  QuaffExpr visitSDivInstruction(Instruction *SDiv, const llvm::SCEV *Expr) {
    assert(SDiv->getOpcode() == Instruction::SDiv &&
           "Assumed SDiv instruction!");

    auto *Dividend = se.getSCEV(SDiv->getOperand(0));
    auto *Divisor = se.getSCEV(SDiv->getOperand(1));
    return visitDivision(Dividend, Divisor, Expr, SDiv);
  }
  QuaffExpr visitDivision(const llvm::SCEV *dividend, const llvm::SCEV *divisor,
                          const llvm::SCEV *S, Instruction *inst = nullptr) {
    // todo
    return QuaffExpr{QuaffClass::Invalid};
  }

  QuaffExpr visitAddRecExpr(const llvm::SCEVAddRecExpr *S) {
    if (!S->isAffine())
      return QuaffExpr{QuaffClass::Invalid};

    const auto start = S->getStart();
    const auto step = S->getStepRecurrence(se);

    if (start->isZero()) {
      auto step = visit(S->getOperand(1));
      return QuaffExpr{QuaffClass::Invalid};
    }

    auto zero_start = se.getAddRecExpr(se.getConstant(start->getType(), 0),
                                       S->getStepRecurrence(se), S->getLoop(),
                                       S->getNoWrapFlags());

    auto res_expr = visit(zero_start);
    auto start_expr = visit(start);

    auto ret_type = std::max(res_expr.type, start_expr.type);

    if (ret_type == QuaffClass::Invalid)
      return QuaffExpr{QuaffClass::Invalid};

    return QuaffExpr{ret_type, res_expr.expr + start_expr.expr};
  }

  QuaffExpr visitSequentialUMinExpr(const llvm::SCEVSequentialUMinExpr *S) {
    // todo
    return QuaffExpr{QuaffClass::Invalid};
  }
  QuaffExpr visitUnknown(const llvm::SCEVUnknown *S) {
    const auto value = S->getValue();
    if (ivdb.isIVar(value)) {
      auto expr =
          fmt::format("{{ {} -> [{}]  }}", indvars, ivdb.getName(value));
      return QuaffExpr{QuaffClass::IVar, islpp::pw_aff{expr}};
    } else
      llvm::dbgs() << "not ivar";

    // todo
    return QuaffExpr{QuaffClass::Invalid};
  }
  QuaffExpr visitCouldNotCompute(const llvm::SCEVCouldNotCompute *S) {
    // todo
    return QuaffExpr{QuaffClass::Invalid};
  }

  template <typename Fn>
  QuaffExpr mergeNary(const llvm::SCEVNAryExpr *S, Fn &&fn) {
    auto val = visit(S->getOperand(0));

    for (int i = 1; i < S->getNumOperands(); ++i) {
      auto inVal = visit(S->getOperand(i));

      auto newClass = std::max(val.type, inVal.type);
      if (newClass == QuaffClass::Invalid)
        return QuaffExpr{QuaffClass::Invalid};

      val = QuaffExpr{newClass, fn(val.expr, inVal.expr)};
    }
    return val;
  }

private:
  RegionInvariants &ri;
  detail::IVarDatabase &ivdb;
  llvm::ScalarEvolution &se;
  string indvars;
};

class PscopBuilder {
public:
  PscopBuilder(RegionInfo &ri, LoopInfo &li, ScalarEvolution &se,
               SyncBlockDetection &sbd)
      : region_info{ri}, loop_info{li}, scalar_evo{se}, sync_blocks{sbd} {}

  const RegionInvariants *get_parent_domain(const Region *reg) {
    return &iteration_domain[reg];
  }

  Optional<QuaffExpr> getQuasiaffineForm(llvm::Value &val,
                                         RegionInvariants &invariants) {
    auto loop = loop_info.getLoopFor(invariants.region->getEntry());
    auto scev = scalar_evo.getSCEVAtScope(&val, loop);
    llvm::dbgs() << "scev dump: ";
    scev->dump();
    QuaffBuilder builder{invariants, invariants.induction_variables,
                         scalar_evo};

    auto ret = builder.visit(scev);
    return ret ? ret : Optional<QuaffExpr>{};
  }

  void build(Function &f) {
    using namespace islpp;
    // detect distribution domain of function
    affinateRegions();
    return;

    // iterate over bbs of function in BFS
    const auto first = &f.getEntryBlock();
    std::queue<llvm::BasicBlock *> queue;
    llvm::DenseSet<llvm::BasicBlock *> visited{first};

    queue.emplace(first);

    while (!queue.empty()) {
      const auto visit = queue.front();
      queue.pop();
      // retrieve iteration domain from enclosing region
      const auto visit_region = region_info.getRegionFor(visit);
      const auto &invariants = iteration_domain[visit_region];

      // domain extracted
      // now we generate our access relations, visiting all loads and stores,
      // separated by sync block
      {
        int index = 0;
        auto bb_name = visit->getName();
        for (auto &sync_block : sync_blocks.iterateSyncBlocks(*visit)) {

          // generate statement instance
          const auto sb_name = fmt::format("{}_{}", bb_name.data(), index++);
          llvm::dbgs() << "\tsb: " << sb_name << "\n";

          for (auto &instr : sync_block) {
            if (const auto store =
                    llvm::dyn_cast_or_null<llvm::StoreInst>(&instr)) {
              // record the write as a union set on the ptr
              // islpp::union_set;
            }

            if (const auto load =
                    llvm::dyn_cast_or_null<llvm::LoadInst>(&instr)) {
              // record the read
              // islpp::union_set;
            }
          }
        }
      }

      // then we enqueue our children if they are not yet enqueued
      for (auto &&elem : llvm::successors(visit)) {
        const auto reg = region_info.getRegionFor(elem);
        if (visited.contains(elem))
          continue;

        queue.push(elem);
        visited.insert(elem);
      }
    }
  }

  void affinateRegions() {
    using namespace islpp;
    llvm::dbgs() << "START AFFINNATE\n";
    region_info.dump();
    loop_info.print(llvm::dbgs());
    const auto top_level_region = region_info.getTopLevelRegion();
    iteration_domain[top_level_region] = ([&]() {
      auto top_level_region_invar = RegionInvariants(top_level_region);
      top_level_region_invar.induction_variables.addThreadIdentifier(
          "llvm.nvvm.read.ptx.sreg.tid.x", "tidx");
      return top_level_region_invar;
    })();

    // bfs traversal
    // though honestly dfs works as well
    stack<Region *> stack;
    stack.push(top_level_region);

    while (!stack.empty()) {
      auto visit = stack.top();
      stack.pop();
      llvm::dbgs() << "visit: " << visit->getNameStr() << "\n";
      auto &parent_invariants = iteration_domain[visit->getParent()];

      if (iteration_domain.find(visit) == iteration_domain.end()) {
        auto invariants = parent_invariants.clone(visit);
        // check our region's constraints
        if (const auto loop = loop_info.getLoopFor(visit->getEntry())) {
          // build loop
          // are our constraints affine?
          if (const auto bounds = loop->getBounds(scalar_evo)) {
            auto &init = bounds->getInitialIVValue();
            auto &fin = bounds->getFinalIVValue();
            const auto step = bounds->getStepValue();

            const auto init_aff = getQuasiaffineForm(init, parent_invariants);
            const auto fin_aff = getQuasiaffineForm(fin, parent_invariants);

            if (init_aff && fin_aff) {
              // if so, add the iteration variable into our iteration domain
              const auto loopVar = loop->getCanonicalInductionVariable();
              invariants.induction_variables.addInductionVariable(loopVar);

              // hacky, but we will reevaluate our init and final exprs to
              // update the expression space
              auto i = getQuasiaffineForm(init, invariants)->expr;
              auto f = getQuasiaffineForm(fin, invariants)->expr;

              // we generate an identity function for the newly added variable
              auto ident = pw_aff{fmt::format(
                  "{{ {} -> [{}]}}", invariants.induction_variables.getVector(),
                  invariants.induction_variables.getName(loopVar))};
              invariants.induction_variables.addConstraint(le_set(i, ident) *
                                                           lt_set(ident, f));

              // [tid] : 0 <= tid <= 255
              // [tid] -> [0], [tid] -> [5]
              // [tid]
              // [tid, i] -> [i]
              // ------------------------------------
              // goal: [tid,i] : 0 <= tid <= 255 and 0 <= i < 5
            }
          } else
            llvm::dbgs() << "no bounds\n";

          // const auto domain =
          //     init_aff && fin_aff
          //         ? lt_map(range(init_aff->expr), range(fin_aff->expr))
          //         : map{"{}"};

          llvm::dbgs() << visit->getNameStr() << " - "
                       << invariants.induction_variables.getDomain() << "\n";
        } else
          llvm::dbgs() << "no loop\n";
        // are we a result of a branch?
        // if so, check if the constraint is affine

        invariants.iteration_domain;

        // then, add the constraint into our set

        iteration_domain[visit] = invariants; // pray for copy ellision
      }

      for (auto &child : *visit) {
        llvm::dbgs() << "push: " << child->getNameStr() << "\n";
        stack.push(child.get());
      }
    }
  }

private:
  DenseMap<const Region *, RegionInvariants> iteration_domain;
  RegionInfo &region_info;
  LoopInfo &loop_info;
  ScalarEvolution &scalar_evo;
  SyncBlockDetection &sync_blocks;
};

AnalysisKey PscopBuilderPass::Key;

PscopBuilderPass::Result PscopBuilderPass::run(Function &f,
                                               FunctionAnalysisManager &fam) {
  PscopBuilder builder{fam.getResult<llvm::RegionInfoAnalysis>(f),
                       fam.getResult<llvm::LoopAnalysis>(f),
                       fam.getResult<llvm::ScalarEvolutionAnalysis>(f),
                       fam.getResult<golly::SyncBlockDetectionPass>(f)};
  builder.build(f);
  return {};
}
} // namespace golly