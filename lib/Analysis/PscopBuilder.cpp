#include <fmt/format.h>
#include <golly/Analysis/PscopBuilder.h>
#include <golly/Analysis/SyncBlockDetection.h>
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
  detail::IVarDatabase ivars;
  int depth = 0;

  RegionInvariants(Region *reg = nullptr) : region{reg} {}
  RegionInvariants clone(Region *child) const {
    auto ret = *this;
    ret.region = child;
    ret.depth += 1;
    return ret;
  }
};

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
  using RegionAnalysis = DenseMap<const Region *, RegionInvariants>;
  using BBAnalysis = DenseMap<const llvm::BasicBlock *, BBInvariants>;

public:
  PscopBuilder(RegionInfo &ri, LoopInfo &li, ScalarEvolution &se,
               DominatorTree &dom_tree, PostDominatorTree &pdt,
               SyncBlockDetection &sbd)
      : region_info{ri}, loop_info{li}, scalar_evo{se},
        sync_blocks{sbd}, dom_tree{dom_tree}, post_dom{pdt} {}

  Optional<QuaffExpr> getQuasiaffineForm(llvm::Value &val,
                                         RegionInvariants &invariants) {
    auto loop = loop_info.getLoopFor(invariants.region->getEntry());
    auto scev = scalar_evo.getSCEVAtScope(&val, loop);
    // llvm::dbgs() << "scev dump: ";
    // scev->dump();
    QuaffBuilder builder{invariants, invariants.ivars, scalar_evo};

    auto ret = builder.visit(scev);
    return ret ? ret : Optional<QuaffExpr>{};
  }

  void build(Function &f) {
    using namespace islpp;
    // detect distribution domain of function
    auto region_analysis = affinateRegions();
    auto bb_analysis = affinateConstraints(f, region_analysis);
    auto domain = buildDomain(f, bb_analysis);
    auto schedule = buildSchedule(f, region_analysis, bb_analysis);
  }

  RegionAnalysis affinateRegions() {
    using namespace islpp;
    llvm::dbgs() << "START AFFINNATE REGIONS\n";

    RegionAnalysis region_analysis;

    region_info.dump();
    loop_info.print(llvm::dbgs());
    const auto top_level_region = region_info.getTopLevelRegion();
    region_analysis[top_level_region] = ([&]() {
      auto top_level_region_invar = RegionInvariants(top_level_region);
      top_level_region_invar.ivars.addThreadIdentifier(
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
      auto &parent_invariants = region_analysis[visit->getParent()];

      if (region_analysis.find(visit) == region_analysis.end()) {
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
              invariants.ivars.addInductionVariable(loopVar);

              // hacky, but we will reevaluate our init and final exprs to
              // update the expression space
              auto i = getQuasiaffineForm(init, invariants)->expr;
              auto f = getQuasiaffineForm(fin, invariants)->expr;

              // we generate an identity function for the newly added variable
              auto ident = pw_aff{
                  fmt::format("{{ {} -> [{}]}}", invariants.ivars.getVector(),
                              invariants.ivars.getName(loopVar))};
              invariants.ivars.addConstraint(le_set(i, ident) *
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
                       << invariants.ivars.getDomain() << "\n";
        } else
          llvm::dbgs() << "no loop\n";

        // then, add the constraint into our set
        region_analysis[visit] = invariants; // pray for copy ellision
      }

      for (auto &child : *visit)
        stack.push(child.get());
    }

    return region_analysis;
  }

  BBAnalysis affinateConstraints(Function &f, RegionAnalysis &region_analysis) {
    llvm::dbgs() << "START AFFINNATE CONSTRAINTS\n";

    BBAnalysis bb_analysis;

    // we generate constraint sets for all ICmpInstructions
    // since we know all indvars, we know which cmps are valid and which are
    // not
    DenseMap<const llvm::ICmpInst *, llvm::BasicBlock *> cmps;
    DenseMap<const llvm::BranchInst *, llvm::BasicBlock *> branches;
    DenseMap<llvm::BasicBlock *, const llvm::BranchInst *> branching_bbs;

    for (auto &bb : f) {
      auto region = region_info.getRegionFor(&bb);
      bb_analysis[&bb] =
          BBInvariants{region_analysis[region].ivars.getDomain()};

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
      auto region = region_info.getRegionFor(bb);
      auto &region_inv = region_analysis[region];

      const auto lhs = getQuasiaffineForm(*icmp->getOperand(0), region_inv);
      const auto rhs = getQuasiaffineForm(*icmp->getOperand(1), region_inv);

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
        llvm::dbgs() << *icmp << " " << constraint << "\n";
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

      for (auto &sb : sync_blocks.iterateSyncBlocks(*bb)) {
        sb.getName();

        auto out = name(domain, sb.getName());
        auto in = name(islpp::set{"{ [] }"}, sb.getName());

        llvm::dbgs() << unwrap(cross(in, out)) << "\n";
      }
    }
    return ret;
  }

  islpp::union_map buildSchedule(Function &f, RegionAnalysis &reg_analysis,
                                 BBAnalysis &bb_analysis) {
    llvm::dbgs() << "START BUILDING SCHEDULE\n";

    auto distribution_domain =
        reg_analysis[region_info.getTopLevelRegion()].ivars.getDomain();

    struct LoopStatus {
      islpp::multi_aff next_map;
    };

    llvm::DenseMap<llvm::Loop *, LoopStatus> loop_setup;
    llvm::DenseMap<golly::SyncBlock *, islpp::multi_aff> analysis;

    queue<llvm::BasicBlock *> queue;
    llvm::DenseSet<llvm::BasicBlock *> visited;
    queue.push(&f.getEntryBlock());
    visited.insert(&f.getEntryBlock());

    // get the parent expression, dispose of the associated
    // Stmt_for_First[tidx] => [0]
    {
      const auto space = get_space(distribution_domain);
      const auto maff = space.zero<islpp::multi_aff>();
      loop_setup.try_emplace(nullptr, LoopStatus{maff});
    }

    while (!queue.empty()) {
      auto visit = queue.front();
      queue.pop();

      auto loop = loop_info.getLoopFor(visit);
      if (auto itr = loop_setup.find(loop); itr == loop_setup.end()) {
        // setup the new loop
        auto parent_loop = loop->getParentLoop();
        assert(loop_setup.find(parent_loop) != loop_setup.end() &&
               "Parent should already be visited");

        auto &loop_status = loop_setup.find(parent_loop)->second;

        loop_setup.try_emplace(
            loop, LoopStatus{append_zero(project_up(loop_status.next_map))});
        loop_status.next_map = increment(loop_status.next_map);
      }

      // now we add ourselves to our domain
      auto &status = loop_setup.find(loop)->second;

      for (auto &sb : sync_blocks.iterateSyncBlocks(*visit)) {
        auto dom = status.next_map;
        dom = islpp::multi_aff{isl_multi_aff_set_tuple_name(
            dom.yield(), isl_dim_type::isl_dim_in, sb.getName().data())};
        analysis.try_emplace(&sb, dom);
        status.next_map = increment(status.next_map);
      }

      for (auto &&child : successors(visit)) {
        if (!visited.contains(child)) {
          queue.push(child);
          visited.insert(child);
        }
      }
    }

    islpp::union_map ret{"{}"};
    for (auto &[bb, dom] : analysis)
      ret = ret + islpp::union_map{islpp::map{dom}};

    llvm::dbgs() << ret << "\n";
    return ret;
  }

  void buildAccessRelations(BBAnalysis &bb_analysis) {

    for (auto &[visit, domain] : bb_analysis) {

      // now we generate our access relations, visiting all loads and stores,
      // separated by sync block
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
            store->getNumOperands();
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
  }

private:
  RegionInfo &region_info;
  LoopInfo &loop_info;
  ScalarEvolution &scalar_evo;
  SyncBlockDetection &sync_blocks;
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
                       fam.getResult<golly::SyncBlockDetectionPass>(f)};
  builder.build(f);
  return {};
}
} // namespace golly