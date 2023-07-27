#ifndef GOLLY_SUPPORT_SCOPHELPER_H
#define GOLLY_SUPPORT_SCOPHELPER_H
#include <llvm/ADT/SetVector.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/ValueHandle.h>

namespace llvm {
class SCEV;
}

namespace polly {

using InvariantLoadsSetTy = llvm::SetVector<llvm::AssertingVH<llvm::LoadInst>>;
using ParameterSetTy = llvm::SetVector<const llvm::SCEV *>;
} // namespace polly

#endif /* GOLLY_SUPPORT_SCOPHELPER_H */
