#include <fmt/format.h>
#include <golly/Analysis/CudaParameterDetection.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/IR/InstVisitor.h>

namespace golly {

using llvm::StringMap;

const static StringMap<Intrinsic> intrinsicLut{
    {"llvm.nvvm.read.ptx.sreg.tid.x",
     {.dim = Dimension::threadX, .type = IntrinsicType::id}},
    {"llvm.nvvm.read.ptx.sreg.tid.y",
     {.dim = Dimension::threadY, .type = IntrinsicType::id}},
    {"llvm.nvvm.read.ptx.sreg.tid.z",
     {.dim = Dimension::threadZ, .type = IntrinsicType::id}},
    {"llvm.nvvm.read.ptx.sreg.ntid.x",
     {.dim = Dimension::threadX, .type = IntrinsicType::count}},
    {"llvm.nvvm.read.ptx.sreg.ntid.y",
     {.dim = Dimension::threadY, .type = IntrinsicType::count}},
    {"llvm.nvvm.read.ptx.sreg.ntid.z",
     {.dim = Dimension::threadZ, .type = IntrinsicType::count}},
    {"llvm.nvvm.read.ptx.sreg.ctaid.x",
     {.dim = Dimension::ctaX, .type = IntrinsicType::id}},
    {"llvm.nvvm.read.ptx.sreg.ctaid.y",
     {.dim = Dimension::ctaY, .type = IntrinsicType::id}},
    {"llvm.nvvm.read.ptx.sreg.ctaid.z",
     {.dim = Dimension::ctaZ, .type = IntrinsicType::id}},
    {"llvm.nvvm.read.ptx.sreg.nctaid.x",
     {.dim = Dimension::ctaX, .type = IntrinsicType::count}},
    {"llvm.nvvm.read.ptx.sreg.nctaid.y",
     {.dim = Dimension::ctaY, .type = IntrinsicType::count}},
    {"llvm.nvvm.read.ptx.sreg.nctaid.z",
     {.dim = Dimension::ctaZ, .type = IntrinsicType::count}},
};

const static std::map<Dimension, string_view> dimensionAliasLut{
    {Dimension::ctaX, "ctaidX"},  {Dimension::ctaY, "ctaidY"},
    {Dimension::ctaX, "ctaidZ"},  {Dimension::threadX, "tidX"},
    {Dimension::threadY, "tidY"}, {Dimension::threadX, "tidZ"},
};

Optional<Intrinsic> CudaParameters::getIntrinsic(const llvm::Value *val) const {
  if (auto itr = detections.find(val); itr != detections.end())
    return itr->second;
  return {};
}

int CudaParameters::getDimensionIndex(Dimension dim) const {
  assert(finalized &&
         "adding new dimensions will change the return value of this function");
  return cached_dim_index.at(dim);
}
int CudaParameters::getCount(Dimension dim) const { return dim_count.at(dim); }

const std::map<Dimension, int> &CudaParameters::getDimCounts() const {
  return dim_count;
}

string_view CudaParameters::getAlias(Dimension dim) const {
  return dimensionAliasLut.at(dim);
}

int CudaParameters::getGridDims() const {
  assert(finalized &&
         "adding new dimensions will change the return value of this function");
  return grid_dim_count;
}

int CudaParameters::getBlockDims() const {
  assert(finalized &&
         "adding new dimensions will change the return value of this function");
  return detections.size() - grid_dim_count;
}

struct IntrinsicFinder : llvm::InstVisitor<IntrinsicFinder> {
  CudaParameters::Builder &builder;

  IntrinsicFinder(CudaParameters::Builder &b) : builder{b} {}

  void visitCall(llvm::CallInst &call) {
    if (const auto fn = call.getCalledFunction()) {
      if (const auto itr = intrinsicLut.find(fn->getName());
          itr != intrinsicLut.end()) {
        builder.addUsage(&call, itr->second);
      }
    }
  }
};

CudaParameters::Builder &
CudaParameters::Builder::addUsage(const llvm::Value *val, Intrinsic in) {
  wip.detections[val] = in;
  addParameter(in);
  return *this;
}

CudaParameters::Builder &CudaParameters::Builder::addParameter(Intrinsic in) {
  if (auto itr = wip.dim_count.find(in.dim); itr == wip.dim_count.end()) {
    // new param
    wip.dim_count[in.dim] = 1; // initialize dim count to 1
  }
  return *this;
}

CudaParameters::Builder &CudaParameters::Builder::addSize(Dimension in,
                                                          int count) {
  wip.dim_count[in] = count;
  return *this;
}

CudaParameters CudaParameters::Builder::build() {
  // count grid dims
  {
    int grid_dims = 0;
    for (auto &[dim, count] : wip.dim_count) {
      if (dim <= Dimension::ctaZ)
        grid_dims++;
      else
        break;
    }

    // if grid is not used, assume grid of (1,1,1)
    if (grid_dims == 0) {
      addParameter(Intrinsic{Dimension::ctaX, IntrinsicType::id});
      grid_dims = 1;
    }
    wip.grid_dim_count = grid_dims;
  }

  // calculate dim index
  {
    int i = 0;
    for (auto &elem : wip.dim_count) {
      wip.cached_dim_index[elem.first] = i++;
    }
  }

  wip.finalized = true;
  return wip;
}

void CudaParameters::dump(llvm::raw_ostream &os) const {
  for (auto [instr, intrin] : detections) {
    os << *instr << " -> " << getAlias(intrin.dim) << "\n";
  }

  if (finalized) {
    for (auto [intrin, index] : cached_dim_index)
      os << fmt::format("[{0}] : {1}", index, getAlias(intrin)) << "\n";
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const CudaParameters &di) {
  di.dump(os);
  return os;
}

AnalysisKey CudaParameterDetection::Key;

CudaParameterDetection::Result
CudaParameterDetection::run(Function &f, FunctionAnalysisManager &am) {
  CudaParameters::Builder b;

  IntrinsicFinder visitor{b};
  visitor.visit(f);

  b.addSize(Dimension::ctaX, 1)
      .addSize(Dimension::threadX, 16)
      .addSize(Dimension::threadY, 16);

  return b.build();
}

PreservedAnalyses CudaParameterDetectionPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  CudaParameterDetection::run(F, AM);
  return PreservedAnalyses::all();
}
} // namespace golly
