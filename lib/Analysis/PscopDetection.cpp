#include <algorithm>
#include <golly/Analysis/DimensionDetection.h>
#include <golly/Analysis/PscopDetection.h>
#include <iostream>
#include <isl/cpp.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/RegionInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/IR/Function.h>

namespace golly {
using llvm::LoopAnalysis;
using llvm::LoopInfo;
using llvm::Region;
using llvm::RegionInfo;
using llvm::RegionInfoAnalysis;
using llvm::ScalarEvolution;
using llvm::ScalarEvolutionAnalysis;
using llvm::SmallDenseSet;

// modified from llvm-project/polly/lib/SCEVValidator.cpp
enum class ScevType {
  Int,
  Param,
  InductionVar,
  Invalid,
};

struct ScevValidator : llvm::SCEVVisitor<ScevValidator, ScevType> {
  ScevType visitConstant(const llvm::SCEVConstant &) { return ScevType::Int; }
};

struct PscopDetection {
  DetectedIntrinsics &dimensions;
  LoopInfo &loop_info;
  RegionInfo &region_info;
  ScalarEvolution &scalar_evolution;

  SmallDenseSet<llvm::Value *> inductionVars;

  PscopDetection(DetectedIntrinsics &dims, ScalarEvolution &scev, LoopInfo &l,
                 RegionInfo &r)
      : dimensions{dims}, loop_info{l}, region_info{r}, scalar_evolution{scev} {
  }

  void run() {
    for (auto &&[instr, type] : dimensions.detections) {
      if (std::ranges::find(threadIds, type) != threadIds.end()) {
        inductionVars.insert(instr);
      }
    }

    findPscops(*region_info.getTopLevelRegion());
  }

  void findPscops(llvm::Region &r) {
    // is this region a branch?

    // is this region a loop?
    if (auto loop = loop_info.getLoopFor(r.getEntry())) {
      llvm::dbgs() << r << " is a loop\n";

      // extract pw_aff bounds
      auto header = loop->getHeader();
      loop->dump();
      llvm::dbgs() << '\n';

      const auto cmp = loop->getLatchCmpInst();
      const auto loopvar = loop->getCanonicalInductionVariable();

      inductionVars.insert(loopvar);

      if (const auto bounds = loop->getBounds(scalar_evolution)) {
        if (isPiecewiseAffine(bounds->getInitialIVValue()) &&
            isPiecewiseAffine(bounds->getFinalIVValue())) {
          llvm::dbgs() << r << " has valid loop bounds";
        }
      }
    }

    for (auto &subregion : r) {
      findPscops(*subregion);
    }
  }

  bool isPiecewiseAffine(llvm::Value &val) {
    if (llvm::isa<llvm::Constant>(val))
      return true;

    if (inductionVars.contains(&val))
      return true;

    return false;
  };
};

PscopDetectionPass::Result
PscopDetectionPass::run(Function &f, FunctionAnalysisManager &am) {
  auto &dd = am.getResult<DimensionDetection>(f);
  auto &se = am.getResult<ScalarEvolutionAnalysis>(f);
  auto &la = am.getResult<LoopAnalysis>(f);
  auto &ri = am.getResult<RegionInfoAnalysis>(f);

  PscopDetection pscops{dd, se, la, ri};
  pscops.run();

  return {};
}

PreservedAnalyses RunPscopDetection::run(Function &f,
                                         FunctionAnalysisManager &am) {
  am.getResult<PscopDetectionPass>(f);
  return PreservedAnalyses::all();
}

} // namespace golly
