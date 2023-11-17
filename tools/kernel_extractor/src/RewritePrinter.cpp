#include "RewritePrinter.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm> // std::all_of
#include <fstream>   // std::ofstream
#include <utility>   // std::move

namespace nus::test {

namespace detail {
bool isCompilableFilepath(std::string_view path) {
  return path.find(".c") != std::string_view::npos ||
         path.find(".cc") != std::string_view::npos ||
         path.find(".cpp") != std::string_view::npos;
}
} // namespace detail

RewritePrinter::RewritePrinter(const RewriteRule &rule)
    : Transformer(rule,
                  [this](llvm::Expected<clang::tooling::AtomicChange> ac) {
                    if (ac)
                      Changes.push_back(std::move(*ac));
                  }) {}

void RewritePrinter::onEndOfTranslationUnit() {
  if (Changes.size() > 0) {
    auto filepath = Changes.front().getFilePath();
    bool SameFilePath = std::ranges::all_of(
        Changes, [&filepath](decltype(Changes)::const_reference ac) -> bool {
          return ac.getFilePath() == filepath;
        });

    if (detail::isCompilableFilepath(filepath) && SameFilePath) {
      // open the file
      if (auto buffer = llvm::MemoryBuffer::getFile(filepath, true)) {
        if (auto patched_code = clang::tooling::applyAtomicChanges(
                filepath, (*buffer)->getBuffer(), Changes, {})) {
          auto patch_file = filepath + ".ext.cu";
          if (auto open_file = std::ofstream{patch_file})
            open_file << *patched_code; // print the code out to ext file
        }
      }
    }
  }
}
} // namespace nus::test