#ifndef GOLLY_SUPPORT_GOLLYOPTIONS_H
#define GOLLY_SUPPORT_GOLLYOPTIONS_H

#include "golly/Analysis/CudaParameterDetection.h"
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/StringMap.h>

namespace golly {

struct dim3 {
  llvm::Optional<int> x, y, z;

  explicit operator bool() const { return x || y || z; }
};

struct GollyOptions {
  struct Params {
    dim3 block, grid;
  };

  llvm::StringMap<Params> function_parameters;

  llvm::Optional<std::string> errorLog;
  bool verboseLog = false;
};

llvm::Expected<GollyOptions> parseOptions(llvm::StringRef name);

} // namespace golly

#endif /* GOLLY_SUPPORT_GOLLYOPTIONS_H */
