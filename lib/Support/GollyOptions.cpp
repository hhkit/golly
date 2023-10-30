#include "golly/Support/GollyOptions.h"

#include <llvm/Support/FormatVariadic.h>

namespace golly {
llvm::Expected<GollyOptions> parseOptions(llvm::StringRef name) {
  return llvm::make_error<llvm::StringError>(
      llvm::formatv("not implemented").str(), llvm::inconvertibleErrorCode());
}
} // namespace golly