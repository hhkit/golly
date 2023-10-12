#ifndef GOLLY_SUPPORT_CONDITIONALATOMIZER_H
#define GOLLY_SUPPORT_CONDITIONALATOMIZER_H

#include <llvm/ADT/SetVector.h>
#include <llvm/IR/Value.h>

namespace golly {
llvm::SetVector<llvm::Value *> atomize(llvm::Value *cond);
} // namespace golly

#endif /* GOLLY_SUPPORT_CONDITIONALATOMIZER_H */
