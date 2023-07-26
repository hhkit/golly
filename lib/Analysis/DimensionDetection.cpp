#include <golly/Analysis/DimensionDetection.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/IR/InstVisitor.h>

namespace golly {

using llvm::StringMap;

const static StringMap<Intrinsic> intrinsicLut{
    {"llvm.nvvm.read.ptx.sreg.tid.x", Intrinsic::tidX},
    {"llvm.nvvm.read.ptx.sreg.tid.y", Intrinsic::tidY},
    {"llvm.nvvm.read.ptx.sreg.tid.z", Intrinsic::tidZ},
    {"llvm.nvvm.read.ptx.sreg.ntid.x", Intrinsic::nTidX},
    {"llvm.nvvm.read.ptx.sreg.ntid.y", Intrinsic::nTidY},
    {"llvm.nvvm.read.ptx.sreg.ntid.z", Intrinsic::nTidZ},
    {"llvm.nvvm.read.ptx.sreg.ctaid.x", Intrinsic::ctaX},
    {"llvm.nvvm.read.ptx.sreg.ctaid.y", Intrinsic::ctaY},
    {"llvm.nvvm.read.ptx.sreg.ctaid.z", Intrinsic::ctaZ},
    {"llvm.nvvm.read.ptx.sreg.nctaid.x", Intrinsic::nCtaX},
    {"llvm.nvvm.read.ptx.sreg.nctaid.y", Intrinsic::nCtaY},
    {"llvm.nvvm.read.ptx.sreg.nctaid.z", Intrinsic::nCtaZ},
};

struct IntrinsicFinder : llvm::InstVisitor<IntrinsicFinder> {
  DetectedIntrinsics detectedIntrinsics;

  void visitCall(llvm::CallInst &call) {
    if (const auto fn = call.getCalledFunction()) {
      if (const auto itr = intrinsicLut.find(fn->getName());
          itr != intrinsicLut.end()) {
        detectedIntrinsics.detections.try_emplace(&call, itr->second);
      }
    }
  }
};

AnalysisKey DimensionDetection::Key;

DimensionDetection::Result
DimensionDetection::run(Function &f, FunctionAnalysisManager &am) {
  IntrinsicFinder visitor;
  visitor.visit(f);

  return visitor.detectedIntrinsics;
}

PreservedAnalyses DimensionDetectionPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  DimensionDetection::run(F, AM);
  return PreservedAnalyses::all();
}
} // namespace golly
