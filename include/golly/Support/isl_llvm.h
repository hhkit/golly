#ifndef GOLLY_SUPPORT_ISL_LLVM_H
#define GOLLY_SUPPORT_ISL_LLVM_H
#include "isl.h"
#include <llvm/Support/raw_ostream.h>

namespace islpp {
// we use argument-based name lookup to ensure that not every value in the scope
// invokes this template
template <typename T>
llvm::raw_ostream &operator<<(llvm::raw_ostream &out, const T &val) {
  osstream stream;
  stream << val;
  out << stream.str();
  return out;
}
} // namespace islpp

#endif /* GOLLY_SUPPORT_ISL_LLVM_H */
