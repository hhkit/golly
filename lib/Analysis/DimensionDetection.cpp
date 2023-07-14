#include <golly/Analysis/DimensionDetection.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/IR/InstVisitor.h>

namespace golly {

const static llvm::StringMap<bool(DetectedIntrinsics::*)> intrinsicLut{
    {"llvm.nvvm.read.ptx.sreg.tid.x", &DetectedIntrinsics::tidX},
    {"llvm.nvvm.read.ptx.sreg.tid.y", &DetectedIntrinsics::tidY},
    {"llvm.nvvm.read.ptx.sreg.tid.z", &DetectedIntrinsics::tidZ},
    {"llvm.nvvm.read.ptx.sreg.ntid.x", &DetectedIntrinsics::ntidX},
    {"llvm.nvvm.read.ptx.sreg.ntid.y", &DetectedIntrinsics::ntidY},
    {"llvm.nvvm.read.ptx.sreg.ntid.z", &DetectedIntrinsics::ntidZ},
    {"llvm.nvvm.read.ptx.sreg.ctaid.x", &DetectedIntrinsics::ctaX},
    {"llvm.nvvm.read.ptx.sreg.ctaid.y", &DetectedIntrinsics::ctaY},
    {"llvm.nvvm.read.ptx.sreg.ctaid.z", &DetectedIntrinsics::ctaZ},
    {"llvm.nvvm.read.ptx.sreg.nctaid.x", &DetectedIntrinsics::nctaX},
    {"llvm.nvvm.read.ptx.sreg.nctaid.y", &DetectedIntrinsics::nctaY},
    {"llvm.nvvm.read.ptx.sreg.nctaid.z", &DetectedIntrinsics::nctaZ},
};

struct IntrinsicFinder : llvm::InstVisitor<IntrinsicFinder> {
  DetectedIntrinsics detectedIntrinsics;

  void visitCall(llvm::CallInst &call) {
    if (const auto fn = call.getCalledFunction()) {
      if (const auto itr = intrinsicLut.find(fn->getName());
          itr != intrinsicLut.end()) {
        const auto flag = itr->second;
        detectedIntrinsics.*flag = true;
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
