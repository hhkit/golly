#include <golly/Analysis/DimensionDetection.h>
#include <golly/Analysis/PscopDetection.h>
#include <iostream>
#include <isl/cpp.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/RegionInfo.h>
#include <llvm/IR/Function.h>

namespace golly {
using llvm::LoopAnalysis;
using llvm::LoopInfo;
using llvm::Region;
using llvm::RegionInfo;
using llvm::RegionInfoAnalysis;

struct PscopDetection {
  DetectedIntrinsics &dimensions;
  LoopInfo &loop_info;
  RegionInfo &region_info;

  PscopDetection(DetectedIntrinsics &dims, LoopInfo &l, RegionInfo &r)
      : dimensions{dims}, loop_info{l}, region_info{r} {}

  void run() {
    // verify the function is
    // verify the grid of the function

    findPscops(*region_info.getTopLevelRegion());
  }

  void findPscops(llvm::Region &r) {
    for (auto &subregion : r) {
      // two kinds of regions: branches and loops
      // if branch, instantiate constraint

      // if loop,
      // ensure affine branch
      // instantiate set with constraint
    }
  }
};

PreservedAnalyses PscopDetectionPass::run(Function &f,
                                          FunctionAnalysisManager &am) {
  PscopDetection pscops{am.getResult<DimensionDetection>(f),
                        am.getResult<LoopAnalysis>(f),
                        am.getResult<RegionInfoAnalysis>(f)};

  pscops.run();

  return PreservedAnalyses::all();
}
} // namespace golly
