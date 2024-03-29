#ifndef GOLLY_ANALYSIS_CUDAPARAMETERDETECTION_H
#define GOLLY_ANALYSIS_CUDAPARAMETERDETECTION_H

#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>
#include <map>
#include <span>
#include <string_view>
#include <variant>

namespace golly {

using llvm::AnalysisInfoMixin;
using llvm::AnalysisKey;
using llvm::AnalysisUsage;
using llvm::ArrayRef;
using llvm::DenseSet;
using llvm::Function;
using llvm::FunctionAnalysisManager;
using llvm::FunctionPass;
using llvm::Optional;
using llvm::PreservedAnalyses;
using llvm::SetVector;
using llvm::SmallDenseMap;
using llvm::SmallVector;
using std::string_view;

enum class IntrinsicType {
  id,
  count,
};

// order matters!
enum class Dimension : short {
  ctaX,
  ctaY,
  ctaZ,
  threadX,
  threadY,
  threadZ,
};

enum class ThreadScope : short {
  block,
  device,
  system,
};

enum class AtomicType : short {
  rmw,
  cas,
};

bool is_grid_dim(Dimension);

struct ThreadId {
  Dimension dim;
};

struct ThreadCount {
  Dimension dim;
};

struct Atomic {
  ThreadScope scope;
  AtomicType type;
  llvm::Value *change;
  llvm::Value *read_val;
};

// using Intrinsic = std::variant<ThreadId, ThreadCount, Atomic>;

struct Intrinsic {
  Dimension dim;
  IntrinsicType type;
};

class CudaParameters {
public:
  struct Builder;

  static inline int warpSize = 32;

  Optional<Intrinsic> getIntrinsic(const llvm::Value *) const;
  int getDimensionIndex(Dimension dim) const;
  int getCount(Dimension dim) const;
  const std::map<Dimension, int> &getDimCounts() const;
  bool isSharedMemoryPtr(const llvm::Value *) const;
  ArrayRef<const llvm::Value *> getSharedMemoryPtrs() const;

  static string_view getAlias(Dimension dim);

  int getGridDims() const;
  int getBlockDims() const;

  void dump(llvm::raw_ostream &) const;

private:
  SmallDenseMap<const llvm::Value *, Intrinsic> detections;
  std::map<Dimension, int> dim_count;
  std::map<Dimension, int> cached_dim_index;
  int grid_dim_count{};

  SetVector<const llvm::Value *> shared_mem_ptrs;

  bool finalized = false;

  friend class Builder;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const CudaParameters &di);

class CudaParameterDetection
    : public AnalysisInfoMixin<CudaParameterDetection> {
public:
  using Result = CudaParameters;
  static AnalysisKey Key;

  Result run(Function &F, FunctionAnalysisManager &AM);

private:
};

class CudaParameterDetectionPass : public CudaParameterDetection {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
} // namespace golly

#endif /* GOLLY_ANALYSIS_CUDAPARAMETERDETECTION_H */
