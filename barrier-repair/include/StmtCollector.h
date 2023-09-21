#ifndef INCLUDE_STMTCOLLECTOR_H
#define INCLUDE_STMTCOLLECTOR_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <set>
#include <span>

namespace nus::test {
class StmtCollector : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  using MatchResult = clang::ast_matchers::MatchFinder::MatchResult;

  StmtCollector(std::vector<std::string> sourceLocs) : sourceLocs{sourceLocs} {}

  void run(const MatchResult &Result) override;

  void onEndOfTranslationUnit() override;

private:
  std::vector<std::string> sourceLocs;
  std::function<bool(const clang::Stmt *)> filter;
  std::set<const clang::Stmt *> stmts;
};

} // namespace nus::test

#endif /* INCLUDE_STMTCOLLECTOR_H */
