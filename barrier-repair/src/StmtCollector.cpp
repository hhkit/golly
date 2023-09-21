#include "StmtCollector.h"

#include "SourceLocation.h"
#include "clang-tidy/utils/LexerUtils.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include <ctre-unicode.hpp>
#include <filesystem>
#include <fstream>

static int patch_counter = 0;
namespace nus::test {
void StmtCollector::run(const MatchResult &Result) {
  if (!filter) {
    std::vector<clang::SourceRange> parsedLocs;

    auto sm = Result.SourceManager;

    std::ranges::transform(
        std::span{sourceLocs}, std::back_inserter(parsedLocs),
        [&](std::string_view str) -> clang::SourceRange {
          if (auto [match, file, l0, c0, l1, c1] =
                  ctre::match<"(.*):(\\d+):(\\d+), line:(\\d+):(\\d+)">(str);
              match) {
            return clang::SourceRange(
                golly::getLocation(sm, file, *golly::to_int(l0),
                                   *golly::to_int(c0)),
                golly::getLocation(sm, file, *golly::to_int(l1),
                                   *golly::to_int(c1)));
          }

          if (auto [match, file, l, c0, c1] =
                  ctre::match<"(.*):(\\d+):(\\d+), col:(\\d+)">(str);
              match) {
            return clang::SourceRange(
                golly::getLocation(sm, file, *golly::to_int(l),
                                   *golly::to_int(c0)),
                golly::getLocation(sm, file, *golly::to_int(l),
                                   *golly::to_int(c1)));
          }

          if (auto [match, file, l, c] = ctre::match<"(.*):(\\d+):(\\d+)">(str);
              match) {
            return clang::SourceRange(golly::getLocation(
                sm, file, *golly::to_int(l), *golly::to_int(c)));
          }

          return {};
        });

    filter = [parsedLocs](const clang::Stmt *stmt) -> bool {
      auto sr = stmt->getSourceRange();
      for (auto &elem : parsedLocs) {
        if (sr.fullyContains(elem))
          return true;
      }
      return false;
    };
  }

  for (auto [id, node] : Result.Nodes.getMap()) {
    if (auto stmt = node.get<clang::Stmt>(); stmt && filter(stmt)) {

      node.dump(llvm::outs(), *Result.Context);
      // add barrier in front of statement

      clang::tooling::AtomicChanges acs;
      {
        clang::tooling::AtomicChange ac(*Result.SourceManager,
                                        stmt->getBeginLoc());
        if (auto err = ac.insert(*Result.SourceManager, stmt->getBeginLoc(),
                                 "{\n__syncthreads();\n");
            !err) {
          acs.emplace_back(ac);
        }
      }
      {
        auto loc = clang::tidy::utils::lexer::getUnifiedEndLoc(
                       *stmt, *Result.SourceManager, {})
                       .getLocWithOffset(1);
        clang::tooling::AtomicChange ac(*Result.SourceManager, loc);
        if (auto err = ac.insert(*Result.SourceManager, loc, "\n}\n"); !err) {
          acs.emplace_back(ac);
        }
      }

      auto sm = Result.SourceManager;
      auto &fm = sm->getFileManager();
      auto file = sm->getFilename(node.getSourceRange().getBegin());
      auto ext = std::filesystem::path(file.str()).extension().generic_string();
      if (auto buf = fm.getBufferForFile(file)) {
        if (auto code = clang::tooling::applyAtomicChanges(
                file, (*buf)->getBuffer(), acs, {})) {

          std::ofstream file{std::to_string(patch_counter++) + ext};
          file << *code;
        }
      }
    }
  }
}

void StmtCollector::onEndOfTranslationUnit() {
  // for (auto &stmt : stmts) {
  //   clang::tooling::AtomicChange ac();
  // }
}
} // namespace nus::test