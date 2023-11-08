#include "golly/Analysis/CudaParameterDetection.h"
#include "golly/Support/GollyOptions.h"
#include "golly/golly.h"
#include <charconv>
#include <ctre-unicode.hpp>
#include <fmt/format.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/IR/InstVisitor.h>
#include <llvm/Support/CommandLine.h>
#include <variant>

#define DEBUG_TYPE "golly-verbose"

namespace golly {

using llvm::StringMap;

bool is_grid_dim(Dimension d) { return d <= Dimension::ctaZ; }

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
    {Dimension::ctaZ, "ctaidZ"},  {Dimension::threadX, "tidX"},
    {Dimension::threadY, "tidY"}, {Dimension::threadZ, "tidZ"},
};

namespace cl = llvm::cl;

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const dim3 &d) {
  if (!d)
    return os << "none";

  if (d.x)
    os << d.x << " ";
  if (d.y)
    os << d.y << " ";
  if (d.z)
    os << d.z << " ";
  return os;
}

class CudaParameters::Builder {
public:
  Builder &addUsage(const llvm::Value *val, Intrinsic in);
  Builder &addParameter(Intrinsic in);
  Builder &addSize(Dimension in, int count);
  Builder &addSharedMemoryPtr(const llvm::Value *val);

  int block_dims_specified() const {
    return std::ranges::count_if(wip.getDimCounts(), [](const auto &v) {
      return !is_grid_dim(v.first);
    });
  }
  int grid_dims_specified() const {
    return std::ranges::count_if(
        wip.getDimCounts(), [](const auto &v) { return is_grid_dim(v.first); });
  }
  CudaParameters build();

private:
  CudaParameters wip;

  friend class CudaParameterDetection;
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

bool CudaParameters::isSharedMemoryPtr(const llvm::Value *ptr) const {
  return shared_mem_ptrs.contains(ptr);
}

ArrayRef<const llvm::Value *> CudaParameters::getSharedMemoryPtrs() const {
  return shared_mem_ptrs.getArrayRef();
}

string_view CudaParameters::getAlias(Dimension dim) {
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

  void visitGetElementPtr(llvm::GetElementPtrInst &inst) {
    auto ptr_operand = inst.getPointerOperand();
    if (llvm::dyn_cast<llvm::GlobalValue>(ptr_operand)) {
      builder.addSharedMemoryPtr(ptr_operand);
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

CudaParameters::Builder &
CudaParameters::Builder::addSharedMemoryPtr(const llvm::Value *val) {
  wip.shared_mem_ptrs.insert(val);
  return *this;
}

CudaParameters CudaParameters::Builder::build() {
  // count grid dims
  {
    int grid_dims = 0;
    for (auto &[dim, count] : wip.dim_count) {
      if (is_grid_dim(dim))
        grid_dims++;
      else
        break;
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

  // for (auto &elem : wip.getSharedMemoryPtrs()) {
  //   llvm::dbgs() << "shmem: " << *elem << "\n";
  // }
  return std::move(wip);
}

void CudaParameters::dump(llvm::raw_ostream &os) const {
  std::vector<int> grid_dims;
  std::vector<int> block_dims;
  for (auto [dim, count] : dim_count)
    if (is_grid_dim(dim))
      grid_dims.emplace_back(count);

  for (auto [dim, count] : dim_count)
    if (!is_grid_dim(dim))
      block_dims.emplace_back(count);

  os << fmt::format("gridDims: [{}], blockDims: [{}]",
                    fmt::join(grid_dims, ","), fmt::join(block_dims, ","));

  LLVM_DEBUG(
      for (auto [instr, intrin]
           : detections) {
        os << *instr << " -> " << getAlias(intrin.dim) << "\n";
      }

      for (auto [dim, count]
           : dim_count) { os << getAlias(dim) << " -> " << count << "\n"; }

      if (finalized) {
        for (auto [intrin, index] : cached_dim_index)
          os << fmt::format("[{0}] : {1}", index, getAlias(intrin)) << "\n";
      });
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const CudaParameters &di) {
  di.dump(os);
  return os;
}

AnalysisKey CudaParameterDetection::Key;

CudaParameterDetection::Result
CudaParameterDetection::run(Function &f, FunctionAnalysisManager &am) {
  CudaParameters::Builder b;

  for (auto &global : f.getParent()->getGlobalList()) {
    if (global.getType()->isPointerTy())
      b.addSharedMemoryPtr(&global);
  }

  IntrinsicFinder visitor{b};
  visitor.visit(f);

  auto opt = RunGollyPass::getOptions();
  auto name = f.getName();

  auto ptr = ([&]() -> golly::GollyOptions::Params * {
    if (!opt)
      return nullptr;
    for (auto &elem : opt->function_parameters)
      if (name.contains(elem.getKey()))
        return &elem.getValue();

    return nullptr;
  })();

  dim3 blockDims = ptr ? ptr->block : dim3{};
  dim3 gridDims = ptr ? ptr->grid : dim3{};

  if (blockDims) {
    // oracle provided
    if (blockDims.x)
      b.addSize(Dimension::threadX, *blockDims.x);
    if (blockDims.y)
      b.addSize(Dimension::threadY, *blockDims.y);
    if (blockDims.z)
      b.addSize(Dimension::threadZ, *blockDims.z);
  } else {
    // no oracle provided
    // making a guess based on intrinsics used
    static int heurestic[][3] = {
        {},
        {256, 1, 1},
        {32, 8, 1},
        {32, 8, 1},
    };

    int c = std::ranges::count_if(b.wip.getDimCounts(), [](const auto &e) {
      return !is_grid_dim(e.first);
    });

    if (c == 0) // no dims detected, assume 1D block
      b.addSize(Dimension::threadX, heurestic[1][0]);
    else {
      int i = 0;
      for (auto &[dim, count] : b.wip.getDimCounts()) {
        if (!is_grid_dim(dim))
          b.addSize(dim, heurestic[c][i++]);
      }
    }
  }

  if (gridDims) {
    // oracle provided
    if (gridDims.x)
      b.addSize(Dimension::ctaX, *gridDims.x);
    if (gridDims.y)
      b.addSize(Dimension::ctaY, *gridDims.y);
    if (gridDims.z)
      b.addSize(Dimension::ctaZ, *gridDims.z);
  } else {
    // no oracle provided
    // making a guess based on intrinsics used
    static int heurestic[][3] = {
        {},
        {1024, 1, 1},
        {32, 32, 1},
        {16, 16, 16},
    };

    int c = std::ranges::count_if(b.wip.getDimCounts(), [](const auto &e) {
      return is_grid_dim(e.first);
    });

    if (c == 0) // no dims detected, assume 1D block
      b.addSize(Dimension::ctaX, heurestic[1][0]);
    else {

      int i = 0;
      for (auto &[dim, count] : b.wip.getDimCounts()) {
        if (is_grid_dim(dim))
          b.addSize(dim, heurestic[c][i++]);
      }
    }
  }

  return b.build();
}

PreservedAnalyses CudaParameterDetectionPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  CudaParameterDetection::run(F, AM);
  return PreservedAnalyses::all();
}
} // namespace golly
