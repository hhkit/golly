#ifndef GOLLY_SUPPORT_GOLLYOPTIONS_H
#define GOLLY_SUPPORT_GOLLYOPTIONS_H

#include "golly/Analysis/CudaParameterDetection.h"
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/StringMap.h>

namespace llvm {
class Function;
}

namespace golly {
struct GollyOptions {
  struct Params {
    std::map<Intrinsic, int> dim_counts;
  };

  llvm::StringMap<Params> function_parameters;
};

llvm::Expected<GollyOptions> parseOptions(llvm::StringRef name);

} // namespace golly

#endif /* GOLLY_SUPPORT_GOLLYOPTIONS_H */
