#ifndef INCLUDE_SOURCELOCATION_H
#define INCLUDE_SOURCELOCATION_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include <charconv>
#include <optional>
#include <string_view>

namespace golly {

inline std::optional<int> to_int(const std::string_view &input) {
  int out;
  const std::from_chars_result result =
      std::from_chars(input.data(), input.data() + input.size(), out);
  if (result.ec == std::errc::invalid_argument ||
      result.ec == std::errc::result_out_of_range) {
    return std::nullopt;
  }
  return out;
}

clang::SourceLocation getLocation(clang::SourceManager *sm,
                                  std::string_view file_name, unsigned line,
                                  unsigned column);
} // namespace golly

#endif /* INCLUDE_SOURCELOCATION_H */
