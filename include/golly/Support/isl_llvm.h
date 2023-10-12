#ifndef GOLLY_SUPPORT_ISL_LLVM_H
#define GOLLY_SUPPORT_ISL_LLVM_H
#include "isl.h"
#include <llvm/Support/raw_ostream.h>

namespace islpp {
// we use argument-based name lookup to ensure that not every value in the scope
// invokes these templates

template <typename T> string to_string(const T &val) {
  osstream stream;
  stream << val;
  return stream.str();
}

template <typename T>
llvm::raw_ostream &operator<<(llvm::raw_ostream &out, const T &val) {
  return out << to_string(val);
}
} // namespace islpp

#define ISLPP_CHECK(VAL)                                                       \
  ([&]() {                                                                     \
    auto v = (VAL);                                                            \
    if (isl_ctx_last_error(islpp::ctx()) != isl_error::isl_error_none) {       \
      llvm::dbgs() << "Error in : " << __FILE__ << ":" << __LINE__ << "\n";    \
      llvm::dbgs() << "  " << isl_ctx_last_error_msg(islpp::ctx()) << "\n";    \
      std::terminate();                                                        \
    }                                                                          \
    return v;                                                                  \
  }())

#endif /* GOLLY_SUPPORT_ISL_LLVM_H */
