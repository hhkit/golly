#include <charconv>
#include <ctre-unicode.hpp>
#include <fmt/format.h>
#include <golly/Analysis/CudaParameterDetection.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/IR/InstVisitor.h>
#include <llvm/Support/CommandLine.h>
#include <variant>

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

struct dim3 {
  Optional<int> x, y, z;

  explicit operator bool() const { return x || y || z; }
};

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
};

struct DimParser : cl::parser<dim3> {
  using cl::parser<dim3>::parser;

  bool parse(cl::Option &O, llvm::StringRef ArgName, llvm::StringRef argValue,
             dim3 &Val) {
    using namespace ctre;

    if (auto [whole, xs, ys, zs] =
            ctre::match<"\\[(\\d+),(\\d+),(\\d+)\\]">(argValue);
        whole) {
      int x, y, z;
      std::from_chars(xs.begin(), xs.end(), x);
      std::from_chars(ys.begin(), ys.end(), y);
      std::from_chars(zs.begin(), zs.end(), z);
      Val.x = x;
      Val.y = y;
      Val.z = z;
      return false;
    };

    if (auto [whole, xs, ys] = ctre::match<"\\[(\\d+),(\\d+)\\]">(argValue);
        whole) {
      int x, y;
      std::from_chars(xs.begin(), xs.end(), x);
      std::from_chars(ys.begin(), ys.end(), y);
      Val.x = x;
      Val.y = y;
      return false;
    };

    if (auto [whole, xs] = ctre::match<"(\\d+)">(argValue); whole) {
      int x;
      std::from_chars(xs.begin(), xs.end(), x);
      Val.x = x;
      return false;
    };
    return false;
  }
};

static cl::opt<dim3, false, DimParser> gridDims("golly-grid-dims",
                                                cl::desc("Grid dimensions"),
                                                cl::value_desc("x | (x,y,z)"));
static cl::opt<dim3, false, DimParser> blockDims("golly-block-dims",
                                                 cl::desc("Block dimensions"),
                                                 cl::value_desc("x | (x,y,z)"));

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
  return std::move(wip);
}

void CudaParameters::dump(llvm::raw_ostream &os) const {
  for (auto [instr, intrin] : detections) {
    os << *instr << " -> " << getAlias(intrin.dim) << "\n";
  }

  for (auto [dim, count] : dim_count) {
    os << getAlias(dim) << " -> " << count << "\n";
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

  // llvm::dbgs() << "block dims: " << blockDims.getValue() << "\n";
  // llvm::dbgs() << "grid dims: " << gridDims.getValue() << "\n";

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
    b.addSize(Dimension::threadX, 32).addSize(Dimension::threadY, 8);
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
    b.addSize(Dimension::ctaX, 1);
  }

  return b.build();
}

PreservedAnalyses CudaParameterDetectionPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  CudaParameterDetection::run(F, AM);
  return PreservedAnalyses::all();
}
} // namespace golly
