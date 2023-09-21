#include "SourceLocation.h"
#include <clang/Basic/FileManager.h>

clang::SourceLocation golly::getLocation(clang::SourceManager *sm,
                                         std::string_view file_name,
                                         unsigned line, unsigned col) {
  if (auto fid = sm->getFileManager().getFileRef(file_name)) {
    return sm->translateFileLineCol(*fid, line, col);
  }
  return {};
}