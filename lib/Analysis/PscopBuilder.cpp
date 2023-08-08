#include <fmt/format.h>
#include <golly/Analysis/PscopBuilder.h>
#include <golly/Analysis/StatementDetection.h>
#include <golly/Support/isl_llvm.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/PriorityQueue.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Analysis/IteratedDominanceFrontier.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/Analysis/RegionInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <queue>
#include <stack>
#include <string>
#include <string_view>
#include <vector>

namespace golly {
using llvm::DenseMap;
using llvm::DenseSet;
using llvm::DominatorTree;
using llvm::LoopInfo;
using llvm::Optional;
using llvm::PostDominatorTree;
using llvm::RegionInfo;
using llvm::ScalarEvolution;
using std::queue;
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

  void addInvariantLoad(const llvm::Value *val) { invariant_loads.insert(val); }

  bool isInvariantLoad(const llvm::Value *val) const {
    return invariant_loads.contains(val);
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
  llvm::DenseSet<const llvm::Value *> invariant_loads;
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

void Pscop::dump(llvm::raw_ostream &os) const {
  os << "domain: " << instantiation_domain << "\n";
  os << "temporal_schedule: " << temporal_schedule << "\n";
  os << "sync_schedule: " << phase_schedule << "\n";
  os << "writes: " << write_access_relation << "\n";
  os << "reads: " << read_access_relation << "\n";
}

using LoopVariables = detail::IVarDatabase;

struct BBInvariants {
  islpp::set domain{"{ [] }"};

  void applyConstraint(islpp::set constraint, bool intersect = true) {
    using namespace islpp;

    auto dim_diff = dims(domain, dim::set) - dims(constraint, dim::set);
    assert(dim_diff >= 0 && "constraint cannot apply to lower loop dimension");
    if (dim_diff > 0)
      constraint = add_dims(constraint, dim::set, dim_diff);

    if (intersect)
      domain = domain * constraint;
    else
      domain = domain - constraint;
  }
};
struct AccessRelations {
  islpp::union_map reads;
  islpp::union_map writes;
};

class QuaffBuilder : public llvm::SCEVVisitor<QuaffBuilder, QuaffExpr> {
public:
  QuaffBuilder(LoopVariables &ri, detail::IVarDatabase &ivdb,
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
      // todo
      auto step = visit(S->getOperand(1));
      // loop MUST exist
      auto indvar = S->getLoop()->getCanonicalInductionVariable();

      if (!indvar) // no canonical indvar
        return QuaffExpr{QuaffClass::Invalid};

      auto loop_expr = islpp::pw_aff{fmt::format(
          "{{ {} -> [{}] }}", ivdb.getVector(), ivdb.getName(indvar))};

      if (step.type != QuaffClass::CInt)
        // loop expr is at least a loop variable, so the step must be const
        // if step is ivar, param, or invalid, all multiplications by indvar are
        // invalid
        return QuaffExpr{QuaffClass::Invalid};

      return QuaffExpr{QuaffClass::IVar, step.expr * loop_expr};
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
    }

    if (ivdb.isInvariantLoad(value)) {
      auto name = value->getName();
      auto expr =
          fmt::format("[{0}] -> {{ {1} -> [{0}]  }}", name.str(), indvars);
      return QuaffExpr{QuaffClass::Param, islpp::pw_aff{expr}};
    }

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
  LoopVariables &ri;
  detail::IVarDatabase &ivdb;
  llvm::ScalarEvolution &se;
  string indvars;
};

class PscopBuilder {
  using LoopInstanceVars = DenseMap<const llvm::Loop *, LoopVariables>;
  using BBAnalysis = DenseMap<const llvm::BasicBlock *, BBInvariants>;

public:
  PscopBuilder(RegionInfo &ri, LoopInfo &li, ScalarEvolution &se,
               DominatorTree &dom_tree, PostDominatorTree &pdt,
               StatementDetection &sbd)
      : region_info{ri}, loop_info{li},
        scalar_evo{se}, stmt_info{sbd}, dom_tree{dom_tree}, post_dom{pdt} {}

  Optional<QuaffExpr> getQuasiaffineForm(llvm::Value &val, llvm::Loop *loop,
                                         LoopVariables &invariants) {
    auto scev = scalar_evo.getSCEVAtScope(&val, loop);
    // llvm::dbgs() << "scev dump: ";
    // scev->dump();
    QuaffBuilder builder{invariants, invariants, scalar_evo};

    auto ret = builder.visit(scev);
    return ret ? ret : Optional<QuaffExpr>{};
  }

  Pscop build(Function &f) {
    using namespace islpp;
    // stmt_info.dump(llvm::dbgs());
    // detect distribution domain of function
    auto loop_analysis = affinateRegions(f);
    auto bb_analysis = affinateConstraints(f, loop_analysis);
    auto instance_domain = buildDomain(f, bb_analysis);
    auto temporal_schedule = buildSchedule(f, loop_analysis, bb_analysis);
    auto access_relations = buildAccessRelations(loop_analysis, bb_analysis);

    return Pscop{
        .instantiation_domain = instance_domain,
        .temporal_schedule = temporal_schedule,
        .phase_schedule = union_map{"{}"},
        .write_access_relation = std::move(access_relations.writes),
        .read_access_relation = std::move(access_relations.reads),
    };
  }

  LoopInstanceVars affinateRegions(Function &f) {
    using namespace islpp;
    LoopInstanceVars loop_to_instances;

    loop_to_instances[nullptr] = ([&]() {
      auto top_level_region_invar = LoopVariables();
      top_level_region_invar.addThreadIdentifier(
          "llvm.nvvm.read.ptx.sreg.tid.x", "tidx");
      for (auto &arg : f.args())
        top_level_region_invar.addInvariantLoad(&arg); // todo: globals

      return top_level_region_invar;
    })();

    bfs(&f.getEntryBlock(), [&](llvm::BasicBlock *visit) {
      auto loop = loop_info.getLoopFor(visit);

      // if new loop
      if (auto itr = loop_to_instances.find(loop);
          itr == loop_to_instances.end()) {
        // setup the new loop
        auto parent_loop = loop->getParentLoop();
        assert(loop_to_instances.find(parent_loop) != loop_to_instances.end() &&
               "Parent should already be visited");

        auto &parent = loop_to_instances.find(parent_loop)->second;
        auto invariants = parent;

        // are our constraints affine?
        if (const auto bounds = loop->getBounds(scalar_evo)) {
          auto &init = bounds->getInitialIVValue();
          auto &fin = bounds->getFinalIVValue();
          const auto step = bounds->getStepValue();

          const auto init_aff = getQuasiaffineForm(init, parent_loop, parent);
          const auto fin_aff = getQuasiaffineForm(fin, parent_loop, parent);

          if (init_aff && fin_aff) {
            // if so, add the iteration variable into our iteration domain
            const auto loopVar = loop->getCanonicalInductionVariable();
            invariants.addInductionVariable(loopVar);

            // hacky, but we will reevaluate our init and final exprs to
            // update the expression space
            auto i = getQuasiaffineForm(init, loop, invariants)->expr;
            auto f = getQuasiaffineForm(fin, loop, invariants)->expr;

            // we generate an identity function for the newly added variable
            auto ident =
                pw_aff{fmt::format("{{ {} -> [{}]}}", invariants.getVector(),
                                   invariants.getName(loopVar))};
            invariants.addConstraint(le_set(i, ident) * lt_set(ident, f));

            // [tid] : 0 <= tid <= 255
            // [tid] -> [0], [tid] -> [5]
            // [tid]
            // [tid, i] -> [i]
            // ------------------------------------
            // goal: [tid,i] : 0 <= tid <= 255 and 0 <= i < 5
          }
        }

        loop_to_instances.try_emplace(loop, invariants);
      }
    });

    return loop_to_instances;
  }

  BBAnalysis affinateConstraints(Function &f, LoopInstanceVars &loop_analysis) {
    BBAnalysis bb_analysis;

    // we generate constraint sets for all ICmpInstructions
    // since we know all indvars, we know which cmps are valid and which are
    // not
    DenseMap<const llvm::ICmpInst *, llvm::BasicBlock *> cmps;
    DenseMap<const llvm::BranchInst *, llvm::BasicBlock *> branches;
    DenseMap<llvm::BasicBlock *, const llvm::BranchInst *> branching_bbs;

    for (auto &bb : f) {
      bb_analysis[&bb] =
          BBInvariants{loop_analysis[loop_info.getLoopFor(&bb)].getDomain()};

      for (auto &instr : bb) {
        if (const auto icmp = llvm::dyn_cast_or_null<llvm::ICmpInst>(&instr))
          cmps.try_emplace(icmp, &bb);
        else if (const auto branch =
                     llvm::dyn_cast_or_null<llvm::BranchInst>(&instr))
          branches.try_emplace(branch, &bb);
      }
    }

    // optimize: forget about the loops
    for (auto &loop : loop_info.getLoopsInPreorder()) {
      cmps.erase(loop->getLatchCmpInst());
      branches.erase(loop->getLoopGuardBranch());
    }

    for (auto &[f, s] : branches)
      branching_bbs.try_emplace(s, f);

    DenseMap<const llvm::ICmpInst *, islpp::set> cmp_constraints;
    for (auto [icmp, bb] : cmps) {
      auto loop = loop_info.getLoopFor(bb);
      auto &region_inv = loop_analysis[loop];

      const auto lhs =
          getQuasiaffineForm(*icmp->getOperand(0), loop, region_inv);
      const auto rhs =
          getQuasiaffineForm(*icmp->getOperand(1), loop, region_inv);

      if (lhs && rhs) {
        const auto constraint = ([&]() -> Optional<islpp::set> {
          switch (icmp->getPredicate()) {
          case llvm::ICmpInst::ICMP_EQ:
            return eq_set(lhs->expr, rhs->expr);
          case llvm::ICmpInst::ICMP_NE:
            return -eq_set(lhs->expr, rhs->expr);
          case llvm::ICmpInst::ICMP_SLT:
            return lt_set(lhs->expr, rhs->expr);
          case llvm::ICmpInst::ICMP_SLE:
            return le_set(lhs->expr, rhs->expr);
          case llvm::ICmpInst::ICMP_SGT:
            return gt_set(lhs->expr, rhs->expr);
          case llvm::ICmpInst::ICMP_SGE:
            return ge_set(lhs->expr, rhs->expr);
          case llvm::ICmpInst::ICMP_ULT:
            return lt_set(lhs->expr, rhs->expr);
          case llvm::ICmpInst::ICMP_ULE:
            return le_set(lhs->expr, rhs->expr);
          case llvm::ICmpInst::ICMP_UGT:
            return gt_set(lhs->expr, rhs->expr);
          case llvm::ICmpInst::ICMP_UGE:
            return ge_set(lhs->expr, rhs->expr);

          default:
            llvm::dbgs() << "unsupported comparison";
            return Optional<islpp::set>{};
          }
        })();

        if (constraint)
          cmp_constraints.try_emplace(icmp, *constraint);
        // llvm::dbgs() << *icmp << " " << constraint << "\n";
      } else
        llvm::dbgs() << "not affine\n";
    }

    // then, we traverse the basic blocks (bfs/dfs both fine) and impose all
    // constraints that dominate a certain bb on that bb to create that basic
    // block's domain
    llvm::SmallVector<llvm::BasicBlock *> pdf;
    for (auto &succeeder : f) {
      llvm::SmallPtrSet<llvm::BasicBlock *, 1> checkMe{&succeeder};
      llvm::ReverseIDFCalculator calc{post_dom};
      calc.setDefiningBlocks(checkMe);
      calc.calculate(pdf);

      // we now have the pdf of succeeder

      // succeeder is control-dependent on all bbs in its post-dominance
      // frontier
      for (auto &pd : pdf) {

        // if any of these bbs are a branch that we are investigating
        if (auto itr = branching_bbs.find(pd); itr != branching_bbs.end()) {
          auto branch = itr->second;
          if (branch->getNumSuccessors() < 2)
            continue;

          const auto cmp =
              llvm::dyn_cast_or_null<llvm::ICmpInst>(branch->getCondition());
          if (!cmp)
            continue;

          if (auto itr = cmp_constraints.find(cmp);
              itr != cmp_constraints.end()) {
            const auto &constraint = itr->second;

            const auto left = branch->getSuccessor(0);
            const auto right = branch->getSuccessor(1);
            auto &analysis = bb_analysis[&succeeder];

            if (dom_tree.dominates(left, &succeeder))
              analysis.applyConstraint(constraint);
            else if (dom_tree.dominates(right, &succeeder))
              analysis.applyConstraint(constraint, false);
          }
        }
      }

      pdf.clear();
    }

    return bb_analysis;
  }

  islpp::union_map buildDomain(Function &f, const BBAnalysis &bb_analysis) {
    islpp::union_map ret{"{}"};
    for (auto &[bb, invar] : bb_analysis) {
      auto domain = invar.domain;

      for (auto &sb : stmt_info.iterateStatements(*bb)) {
        sb.getName();

        auto out = name(domain, sb.getName());
        auto in = name(islpp::set{"{ [] }"}, sb.getName());

        ret = ret + islpp::union_map{unwrap(cross(in, out))};
      }
    }
    return ret;
  }

  islpp::union_map buildSchedule(Function &f, LoopInstanceVars &reg_analysis,
                                 BBAnalysis &bb_analysis) {
    llvm::dbgs() << "START BUILDING SCHEDULE\n";

    auto distribution_domain = reg_analysis[nullptr].getDomain();

    using LoopStatus = islpp::multi_aff;

    llvm::DenseMap<llvm::Loop *, LoopStatus> loop_setup;
    llvm::DenseMap<golly::Statement *, islpp::multi_aff> analysis;

    // get the parent expression, dispose of the associated
    // Stmt_for_First[tidx] => [0]
    {
      const auto space = get_space(distribution_domain);
      const auto maff = space.zero<islpp::multi_aff>();
      loop_setup.try_emplace(nullptr, LoopStatus{maff});
    }

    bfs(&f.getEntryBlock(), [&](llvm::BasicBlock *visit) {
      auto loop = loop_info.getLoopFor(visit);
      if (auto itr = loop_setup.find(loop); itr == loop_setup.end()) {
        // setup the new loop
        auto parent_loop = loop->getParentLoop();
        assert(loop_setup.find(parent_loop) != loop_setup.end() &&
               "Parent should already be visited");

        auto &loop_status = loop_setup.find(parent_loop)->second;

        loop_setup.try_emplace(
            loop, LoopStatus{append_zero(project_up(loop_status))});
        loop_status = increment(loop_status);
      }

      // now we add ourselves to our domain
      auto &status = loop_setup.find(loop)->second;

      for (auto &sb : stmt_info.iterateStatements(*visit)) {
        auto dom = status;
        dom = islpp::multi_aff{isl_multi_aff_set_tuple_name(
            dom.yield(), isl_dim_type::isl_dim_in, sb.getName().data())};
        analysis.try_emplace(&sb, dom);
        status = increment(status);
      }
    });

    islpp::union_map ret{"{}"};
    for (auto &[bb, dom] : analysis)
      ret = ret + islpp::union_map{islpp::map{dom}};
    return ret;
  }

  AccessRelations buildAccessRelations(LoopInstanceVars &loop_analysis,
                                       BBAnalysis &bb_analysis) {
    islpp::union_map writes{"{}"};
    islpp::union_map reads{"{}"};

    for (auto &[visit, domain] : bb_analysis) {

      // now we generate our access relations, visiting all loads and stores,
      // separated by sync block
      auto loop = loop_info.getLoopFor(visit);
      auto &loop_domain = loop_analysis[loop];

      for (auto &statement : stmt_info.iterateStatements(*visit)) {
        if (const auto mem_access =
                statement.as<golly::MemoryAccessStatement>()) {
          auto ptr_operand =
              const_cast<llvm::Value *>(mem_access->getPointerOperand());

          if (auto qa = getQuasiaffineForm(*ptr_operand, loop, loop_domain)) {
            auto as_map =
                name(islpp::map{qa->expr}, islpp::dim::in, statement.getName());
            switch (mem_access->getAccessType()) {
            case MemoryAccessStatement::Access::Read:
              reads = reads + islpp::union_map{as_map};
              break;
            case MemoryAccessStatement::Access::Write:
              writes = writes + islpp::union_map{as_map};
              break;
            }
          }
        }
      }
    }

    return {
        .reads = std::move(reads),
        .writes = std::move(writes),
    };
  }

  template <typename T> void bfs(llvm::BasicBlock *first, T &&fn) {

    queue<llvm::BasicBlock *> queue;
    llvm::DenseSet<llvm::BasicBlock *> visited;
    queue.push(first);
    visited.insert(first);

    while (!queue.empty()) {
      auto visit = queue.front();
      queue.pop();

      fn(visit);

      for (auto &&child : successors(visit)) {
        if (!visited.contains(child)) {
          queue.push(child);
          visited.insert(child);
        }
      }
    }
  }

private:
  RegionInfo &region_info;
  LoopInfo &loop_info;
  ScalarEvolution &scalar_evo;
  StatementDetection &stmt_info;
  DominatorTree &dom_tree;
  PostDominatorTree &post_dom;
};

AnalysisKey PscopBuilderPass::Key;

PscopBuilderPass::Result PscopBuilderPass::run(Function &f,
                                               FunctionAnalysisManager &fam) {
  PscopBuilder builder{fam.getResult<llvm::RegionInfoAnalysis>(f),
                       fam.getResult<llvm::LoopAnalysis>(f),
                       fam.getResult<llvm::ScalarEvolutionAnalysis>(f),
                       fam.getResult<llvm::DominatorTreeAnalysis>(f),
                       fam.getResult<llvm::PostDominatorTreeAnalysis>(f),
                       fam.getResult<golly::StatementDetectionPass>(f)};
  auto ret = builder.build(f);
  ret.dump(llvm::dbgs());
  return ret;
}
} // namespace golly