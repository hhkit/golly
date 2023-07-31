#include <golly/Analysis/PscopBuilder.h>
#include <golly/Support/isl_llvm.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/PriorityQueue.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/RegionInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/IR/Instructions.h>
#include <queue>

namespace golly {
using llvm::LoopInfo;
using llvm::Optional;
using llvm::RegionInfo;
using llvm::ScalarEvolution;

using InductionVarSet = llvm::DenseSet<const llvm::Value *>;

namespace detail {
bool isInductionVariable(
    const llvm::Value *checkme, const InductionVarSet &iteration_vars,
    const llvm::StringSet</*nani???*/> &distribution_vars) {
  if (iteration_vars.contains(checkme))
    return true;

  if (auto callinst = llvm::dyn_cast_or_null<llvm::CallInst>(checkme)) {
    const auto fn_name = callinst->getFunction()->getName();
    return distribution_vars.contains(fn_name);
  }

  return false;
}

} // namespace detail

struct QuasiaffineForm {};

class PscopBuilder {
public:
  struct RegionInvariants {
    isl_set *iteration_domain;
    InductionVarSet induction_variables;
  };

  PscopBuilder(RegionInfo &ri, LoopInfo &li, ScalarEvolution &se)
      : region_info{ri}, loop_info{li}, scalar_evo{se} {}

  const RegionInvariants *get_parent_domain(const Region *reg) {
    if (reg == nullptr)
      return {};

    // assert((iteration_domain.find(reg) != iteration_domain.end()) &&
    //        "parent should already have been visited");
    return &iteration_domain[reg];
  }

  Optional<QuasiaffineForm> getQuasiaffineForm(const llvm::Value &val,
                                               RegionInvariants &invariants) {
    *val.getValueName();
    return {};
  }

  // returns the quasiaffine form of the constraint if the bb is the result of
  // an if comparison note: negate the constraint if it is the else path!
  Optional<QuasiaffineForm>
  getQuasiaffineConstraint(const llvm::BasicBlock *bb,
                           RegionInvariants &invariants) {
    return {};
  }

  void build(Function &f) {
    using namespace islpp;
    // detect distribution domain of function
    // auto distribution_domain = islpp::set("{[tid]       | 0 <= 2 * tid <= 255
    // and 32 <= tid}");
    auto test = union_set(" { A[tid] | 0 <= tid <= 255}");
    auto test2 = union_set(" { B[tid] | 0 <= tid <= 255}");
    auto test3 = test + test2;
    // auto set1 = set("{ B[0]; A[1] }"); // fails because set has more than one
    // tuple

    auto deflt = union_set();
    auto dom1 = union_set("{ B[0]; A[2,8,1] }");
    auto dom2 = union_set(" { A[2, 8, 1]}");
    auto dom3 = dom1 - dom2;

    // do not reuse variables used in other equtions
    auto dom4 = dom1 * deflt;
    // auto dom2 = islpp::set("{  B[0]; A[2,8,1] }");

    llvm::dbgs() << "isl: " << dom3 << "\n";
    llvm::dbgs() << "lexmax: " << lexmax(test3) << "\n";

    // iterate over bbs of function in BFS
    const auto first = &f.getEntryBlock();
    std::queue<llvm::BasicBlock *> queue;
    queue.emplace(first);

    llvm::DenseSet<llvm::BasicBlock *> visited{first};

    while (!queue.empty()) {
      const auto visit = queue.front();
      queue.pop();
      // retrieve iteration domain from enclosing region
      // note:
      // by induction, we will have subsequent enclosing regions'
      // iteration domains as well
      const auto visit_region = region_info.getRegionFor(visit);
      auto parent_invariants = get_parent_domain(visit_region->getParent());

      // has this region been analyzed yet?
      // note: regions only need to be visited once
      // but because WARs and RARs may occur after deeper regions
      // we still need to visit the basic blocks in sequence to generate a
      // lexicographically correct timing
      if (iteration_domain.find(visit_region) != iteration_domain.end()) {
        auto indvars =
            parent_invariants ? *parent_invariants : RegionInvariants{};
        // check our region's constraints
        // are we a loop?
        if (const auto visit_loop =
                loop_info.getLoopFor(visit_region->getEntry())) {
          // are our constraints affine?
          const auto bounds = visit_loop->getBounds(scalar_evo);
          const auto &init = bounds->getInitialIVValue();
          const auto &fin = bounds->getFinalIVValue();

          // if so, add the iteration variable into our iteration domain
          const auto loopVar = visit_loop->getCanonicalInductionVariable();
          indvars.induction_variables.insert(loopVar);
        }

        // are we a result of a branch?
        // if so, check if the constraint is affine

        indvars.iteration_domain;

        // then, add the constraint into our set

        iteration_domain[visit_region] =
            std::move(indvars); // pray for copy ellision
      }

      // now we generate our access relations, visiting all loads and stores,
      // separated by sync block

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

private:
  DenseMap<const Region *, RegionInvariants> iteration_domain;
  RegionInfo &region_info;
  LoopInfo &loop_info;
  ScalarEvolution &scalar_evo;
};

AnalysisKey PscopBuilderPass::Key;

PscopBuilderPass::Result PscopBuilderPass::run(Function &f,
                                               FunctionAnalysisManager &fam) {
  PscopBuilder builder{fam.getResult<llvm::RegionInfoAnalysis>(f),
                       fam.getResult<llvm::LoopAnalysis>(f),
                       fam.getResult<llvm::ScalarEvolutionAnalysis>(f)};
  builder.build(f);
  return {};
}
} // namespace golly