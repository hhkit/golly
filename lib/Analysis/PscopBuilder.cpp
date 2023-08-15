#include <fmt/format.h>
#include <golly/Analysis/PscopBuilder.h>
#include <golly/Analysis/StatementDetection.h>
#include <golly/Support/Traversal.h>
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
#include <llvm/IR/InstVisitor.h>
#include <llvm/IR/Instructions.h>
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

using std::string;
using std::string_view;
using std::vector;

using InductionVarSet = llvm::MapVector<const llvm::Value *, string>;

// calculate based on launch parameters
struct FunctionInvariants {
  struct Intrinsic {
    string llvm_symbol;
    string alias;
    islpp::set domain;
  };
  vector<Intrinsic> intrinsics;

  islpp::set distribution_domain;       // the domain of active threads
  islpp::pw_aff getWarpId;              // [tid] -> [wid]
  islpp::pw_aff getLaneId;              // [tid] -> lid
  islpp::multi_pw_aff getWarpLaneTuple; // [tid] -> [wid, lid]
  islpp::map warpTupleToLane;           // [wid, lid] -> [ lid ]
};

namespace detail {
class IVarDatabase {
public:
  void addFunctionAnalysis(const FunctionInvariants &fn_analysis) {
    for (auto &elem : fn_analysis.intrinsics)
      thread_ids.try_emplace(elem.llvm_symbol, elem.alias);

    domain = fn_analysis.distribution_domain;
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

islpp::set mask_to_lane_set(unsigned mask) {
  islpp::set lanes{"{ [lid] : 0 <= lid <= 31 }"};
  for (unsigned i = 0; i < 32; ++i) {
    if ((mask & (1U << i)) == 0) { // bitmask disabled
      lanes = lanes - islpp::set{fmt::format("{{[ {} ]}}", i)};
    }
  }
  return lanes;
}

} // namespace detail

struct OracleData {
  int ntid = 256;    // size of threadblock
  int ncta = 1;      // size of grid
  int warpSize = 32; // size of warp
};

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

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Pscop &pscop) {
  os << "domain:\n  " << pscop.instantiation_domain << "\n";
  os << "distribution_schedule:\n  " << pscop.distribution_schedule << "\n";
  os << "temporal_schedule:\n  " << pscop.temporal_schedule << "\n";
  os << "sync_schedule:\n  " << pscop.sync_schedule << "\n";
  os << "writes:\n  " << pscop.write_access_relation << "\n";
  os << "reads:\n  " << pscop.read_access_relation << "\n";
  return os;
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
        // loop expr is at least a loop variable, so the step must be
        // const if step is ivar, param, or invalid, all multiplications
        // by indvar are invalid
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

class MaskValidator
    : public llvm::SCEVVisitor<MaskValidator, Optional<islpp::union_map>> {
public:
  using Base = llvm::InstVisitor<MaskValidator, Optional<islpp::union_map>>;
  using Result = Optional<islpp::union_map>;

  MaskValidator(islpp::union_map active_threads, islpp::union_map get_lane_id)
      : active_threads{std::move(active_threads)}, get_lane_id{std::move(
                                                       get_lane_id)} {}

  Result visitConstant(const llvm::SCEVConstant *cint) {
    auto mask = cint->getAPInt().getZExtValue();
    // validate the mask before returning

    const auto masks = islpp::union_set{detail::mask_to_lane_set(mask)};

    auto lanes = apply_range(active_threads, get_lane_id); // StmtInst -> lids
    auto masked_lanes = range_intersect(lanes, masks);     // StmtInst -> lids

    if (range(masked_lanes) > range(active_threads)) {
      llvm::dbgs() << "thread specified unreachable with mask in __syncwarps\n";
      return {};
    };

    return islpp::domain_intersect(
        active_threads,
        domain(masked_lanes)); // keep the lanes that are in the masks
  }

  Result visitAddExpr(const llvm::SCEVAddExpr *S) { return {}; }

  Result visitMulExpr(const llvm::SCEVMulExpr *S) { return {}; }

  Result visitPtrToIntExpr(const llvm::SCEVPtrToIntExpr *S) { return {}; }
  Result visitTruncateExpr(const llvm::SCEVTruncateExpr *S) { return {}; }
  Result visitZeroExtendExpr(const llvm::SCEVZeroExtendExpr *S) { return {}; }
  Result visitSignExtendExpr(const llvm::SCEVSignExtendExpr *S) { return {}; }
  Result visitSMaxExpr(const llvm::SCEVSMaxExpr *S) { return {}; }
  Result visitUMaxExpr(const llvm::SCEVUMaxExpr *S) { return {}; }
  Result visitSMinExpr(const llvm::SCEVSMinExpr *S) { return {}; }
  Result visitUMinExpr(const llvm::SCEVUMinExpr *S) { return {}; }

  Result visitUDivExpr(const llvm::SCEVUDivExpr *S) { return {}; }
  Result visitSDivInstruction(Instruction *SDiv, const llvm::SCEV *Expr) {
    return {};
  }
  Result visitDivision(const llvm::SCEV *dividend, const llvm::SCEV *divisor,
                       const llvm::SCEV *S, Instruction *inst = nullptr) {
    return {};
  }

  Result visitAddRecExpr(const llvm::SCEVAddRecExpr *S) { return {}; }

  Result visitSequentialUMinExpr(const llvm::SCEVSequentialUMinExpr *S) {
    return {};
  }
  Result visitUnknown(const llvm::SCEVUnknown *S) { return {}; }
  Result visitCouldNotCompute(const llvm::SCEVCouldNotCompute *S) { return {}; }

private:
  islpp ::union_map active_threads; // StmtInst -> tid
  islpp::union_map get_lane_id;
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

  // take in value by non-const because the SCEV guy doesn't know what const
  // correctness is
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
    auto fn_analysis = detectFunctionInvariants(f);
    auto loop_analysis = affinateRegions(f, fn_analysis);
    auto bb_analysis = affinateConstraints(f, loop_analysis);
    auto statement_domain = buildDomain(f, bb_analysis);
    auto distribution_schedule = buildDistributionSchedule(
        statement_domain, fn_analysis, loop_analysis, bb_analysis);

    auto temporal_schedule = buildSchedule(f, fn_analysis, statement_domain,
                                           loop_analysis, bb_analysis);
    auto sync_schedule = buildSynchronizationSchedule(
        f, fn_analysis, statement_domain, distribution_schedule,
        temporal_schedule, bb_analysis);
    auto access_relations = buildAccessRelations(loop_analysis, bb_analysis);

    return Pscop{
        .instantiation_domain = coalesce(statement_domain),
        .distribution_schedule = coalesce(distribution_schedule),
        .temporal_schedule = coalesce(temporal_schedule),
        .sync_schedule = coalesce(sync_schedule),
        .write_access_relation = coalesce(access_relations.writes),
        .read_access_relation = coalesce(access_relations.reads),
    };
  }

  FunctionInvariants detectFunctionInvariants(Function &f) {
    OracleData oracle;

    auto intrinsics = vector<FunctionInvariants::Intrinsic>{
        {"llvm.nvvm.read.ptx.sreg.tid.x", "tidx",
         islpp::set{fmt::format("{{ [{0}] : 0 <= {0} < {1} }}", "tidx",
                                oracle.ntid)}}};

    islpp::set domain{" { [] } "};

    for (auto &elem : intrinsics)
      domain = flat_cross(domain, elem.domain);

    auto warpid_getter = islpp::pw_aff{fmt::format(
        "{{ [{0}] -> [ floor( {0} / {1} ) ] }} ", "tidx", oracle.warpSize)};
    auto lane_id_getter = islpp::pw_aff{
        fmt::format("{{ [{0}] -> [{0} mod {1}] }} ", "tidx", oracle.warpSize)};

    auto warp_tuple_getter = islpp::multi_pw_aff{
        fmt::format("{{ [{0}] -> [ [floor( {0} / {1} )] -> [{0} mod {1}] ] }} ",
                    "tidx", oracle.warpSize)};

    auto warp_to_lane = islpp::map{islpp::multi_pw_aff{
        fmt::format("{{ [{0}, {1}] -> [ {1} ] }} ", "wid", "lid")}};

    return FunctionInvariants{.intrinsics = intrinsics,
                              .distribution_domain = domain,
                              .getWarpId = warpid_getter,
                              .getLaneId = lane_id_getter,
                              .getWarpLaneTuple = warp_tuple_getter,
                              .warpTupleToLane = warp_to_lane};
  }

  LoopInstanceVars affinateRegions(Function &f,
                                   FunctionInvariants &fn_analysis) {
    using namespace islpp;
    LoopInstanceVars loop_to_instances;

    loop_to_instances[nullptr] = ([&]() {
      auto top_level_region_invar = LoopVariables();
      top_level_region_invar.addFunctionAnalysis(fn_analysis);

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
        auto out = name(domain, sb.getName());
        auto in = name(islpp::set{"{ [] }"}, sb.getName());

        ret = ret + islpp::union_map{unwrap(cross(in, out))};
      }
    }
    return ret;
  }

  islpp::union_map buildDistributionSchedule(
      islpp::union_map statement_domain, const FunctionInvariants &fn_analysis,
      LoopInstanceVars &liv, const BBAnalysis &bb_analysis) {
    islpp::union_map ret{"{}"};

    auto distribution_domain = fn_analysis.distribution_domain;

    const auto distri_dims = dims(distribution_domain, islpp::dim::set);
    for (auto &[bb, invar] : bb_analysis) {
      const auto new_dims = dims(invar.domain, islpp::dim::in);

      auto test = project_onto(invar.domain, islpp::dim::set, 0, distri_dims);

      llvm::dbgs() << "DBG: " << bb->getName() << " - " << test << "\n";
      for (auto &stmt : stmt_info.iterateStatements(*bb)) {
        ret =
            ret + islpp::union_map(name(test, islpp::dim::in, stmt.getName()));
      }
    }

    // restrict to the invocations that actually exist
    ret = domain_intersect(ret, range(statement_domain));
    return ret;
  }

  islpp::union_map buildSchedule(Function &f, FunctionInvariants &fn_analysis,
                                 islpp::union_map statement_domain,
                                 LoopInstanceVars &reg_analysis,
                                 BBAnalysis &bb_analysis) {
    using LoopStatus = islpp::multi_aff;
    const auto &distribution_domain = fn_analysis.distribution_domain;

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

    // trim invalid values
    return domain_intersect(ret, range(statement_domain));
  }

  AccessRelations buildAccessRelations(LoopInstanceVars &loop_analysis,
                                       BBAnalysis &bb_analysis) {
    islpp::union_map writes{"{}"};
    islpp::union_map reads{"{}"};

    for (auto &[visit, domain] : bb_analysis) {

      // now we generate our access relations, visiting all loads and
      // stores, separated by sync block
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

  islpp::union_map buildSynchronizationSchedule(
      Function &f, const FunctionInvariants &fn_analysis,
      islpp::union_map statement_domain, islpp::union_map distribution_schedule,
      islpp::union_map temporal_schedule, BBAnalysis &bb_analysis) {
    islpp::union_map ret{"{}"};

    const auto launched_threads =
        islpp::union_set{fn_analysis.distribution_domain};
    const auto tid_to_stmt_inst = reverse(distribution_schedule);

    for (auto &[visit, _domain] : bb_analysis) {
      auto loop = loop_info.getLoopFor(visit);

      for (auto &statement : stmt_info.iterateStatements(*visit)) {
        const auto stmt_set =
            islpp::union_set{name(islpp::set("{ [] }"), statement.getName())};

        // the statement invocations in this domain
        const auto stmt_domain = apply(stmt_set, statement_domain);

        // threads to statement invocation, limited to this statement
        const auto tid_to_stmt = range_intersect(tid_to_stmt_inst, stmt_domain);

        // if we are a barrier...
        if (const auto barrier = statement.as<golly::BarrierStatement>()) {
          // do barrier stuff

          // retrieve the active threads in this domain
          const auto active_threads = apply(stmt_domain, distribution_schedule);

          if (is_empty(active_threads)) {
            llvm::outs() << "warning: unreachable barrier\n";
            continue;
          }

          // all threads that participate in this statement
          const auto thread_pairs =
              unwrap(cross(active_threads, active_threads)) -
              identity(active_threads);

          // verify that all threads that reach this barrier CAN reach
          // this barrier
          auto barrier_statements = std::visit(
              [&](const auto &bar) -> islpp::union_map {
                using T = std::decay_t<decltype(bar)>;
                if constexpr (std::is_same_v<BarrierStatement::End, T>) {
                  auto dmn_insts =
                      apply_range(domain_map(thread_pairs), tid_to_stmt) +
                      apply_range(range_map(thread_pairs), tid_to_stmt);
                  return coalesce(dmn_insts);
                }

                if constexpr (std::is_same_v<BarrierStatement::Block, T>) {
                  // if it is a block-level barrier
                  const BarrierStatement::Block &block_bar = bar;

                  // every thread in the block must pass through this
                  // barrier, no exceptions

                  // thus, ensure that the stmt distribution domain
                  // encapsulates all threads in the block
                  if (launched_threads == active_threads) {
                    auto dmn_insts =
                        apply_range(domain_map(thread_pairs), tid_to_stmt) +
                        apply_range(range_map(thread_pairs), tid_to_stmt);
                    return dmn_insts;
                  } else {
                    llvm::outs() << "unreachable lanes in __syncthreads, "
                                    "potential branch divergence detected\n"
                                 << launched_threads - active_threads
                                 << " are unreachable.\n";
                    return islpp::union_map{"{}"};
                  }
                }

                if constexpr (std::is_same_v<BarrierStatement::Warp, T>) {
                  // if it is a block-level barrier
                  const BarrierStatement::Warp &warp_bar = bar;
                  // ensure that the threads that the warp is waiting for
                  // can arrive at the warp

                  const auto warp_lane_expr = islpp::union_map{
                      islpp::map{fn_analysis.getWarpLaneTuple}};

                  // retrieve thread id from warp_lane tuple
                  const auto warp_lane_inv_expr = reverse(warp_lane_expr);

                  const auto warp_expr =
                      islpp::union_map{islpp::map{fn_analysis.getWarpId}};

                  // take note:
                  // (wid1 -> lid1) -> (wid2 -> lid2)
                  const auto wl_pairs =
                      apply_range(apply_domain(thread_pairs, warp_lane_expr),
                                  warp_lane_expr);

                  const auto active_warps =
                      apply(active_threads, warp_lane_expr);

                  // so now we can determine threads which are in the same warp

                  // retain a thread pair only if:
                  //  1. threads share the same warp
                  // wl pair is guaranteed to be [i -> j]
                  // therefore, filter out all wl pairs with
                  // [wid1 -> lid1] -> [wid2 -> lid2] s.t. wid1 != wid2

                  // also we can totally hardcode this for now
                  const auto different_warps =
                      islpp::union_map{"{[[wid1] -> [lid1]] -> [[wid2] -> "
                                       "[lid2]] : wid1 != wid2 and 0 <= wid1 < "
                                       "8 and 0 <= wid2 < 8 and 0 <= lid1 < 32 "
                                       "and 0 <= lid2 < 32}"};

                  const auto wls_same_warp = wl_pairs - different_warps;
                  const auto wl_to_tid = reverse(warp_lane_expr);
                  const auto threads_same_warp = apply_range(
                      apply_domain(wls_same_warp, wl_to_tid), wl_to_tid);

                  //  2. the mask accepts both lanes
                  // note that mask is instanced on statements, not enough to go
                  // by mask only

                  // get all threads involved with this statement
                  const auto stmt_distribution =
                      domain_intersect(distribution_schedule, stmt_domain);

                  // StmtInst -> tid
                  const auto waiting_lanes =
                      MaskValidator{
                          stmt_distribution,
                          islpp::union_map{islpp::map{fn_analysis.getLaneId}}}
                          .visit(scalar_evo.getSCEV(warp_bar.mask));

                  if (!waiting_lanes) {
                    // unable to get mask
                    llvm::dbgs() << "non-affine mask subset\n";
                    return islpp::union_map{"{}"};
                  }

                  auto threads_to_lanes = reverse(*waiting_lanes);
                  auto waiting_threads = domain(threads_to_lanes);

                  auto waiting_thread_pairs =
                      unwrap(cross(waiting_threads, waiting_threads)) -
                      identity(waiting_threads);

                  auto threads_to_stmt =
                      apply_range(domain_map(waiting_thread_pairs),
                                  threads_to_lanes) +
                      apply_range(range_map(waiting_thread_pairs),
                                  threads_to_lanes);

                  return domain_intersect(threads_to_stmt,
                                          wrap(threads_same_warp));
                }

                return islpp::union_map{"{}"};
              },
              barrier->getBarrier());

          ret = ret + barrier_statements;
        }
      }
    }

    return ret;
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

  if (f.getName() == "_Z10__syncwarpj") {
    return Pscop{
        .instantiation_domain = islpp::union_map{"{}"},
        .distribution_schedule = islpp::union_map{"{}"},
        .temporal_schedule = islpp::union_map{"{}"},
        .sync_schedule = islpp::union_map{"{}"},
        .write_access_relation = islpp::union_map{"{}"},
        .read_access_relation = islpp::union_map{"{}"},
    };
  }

  auto ret = builder.build(f);
  // llvm::dbgs() << "pscop for " << f.getName() << ": {\n" << ret << "}\n";
  return ret;
}

} // namespace golly