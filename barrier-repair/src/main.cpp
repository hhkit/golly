#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"        // MatchFinder
#include "clang/Tooling/Transformer/RangeSelector.h" // RangeSelector
#include "clang/Tooling/Transformer/Stencil.h"       // cat

#include "StmtCollector.h"
#include <ctre-unicode.hpp>
#include <fstream>
#include <string>

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("repairer options");
static llvm::cl::extrahelp
    CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);

static llvm::cl::opt<std::string> locs("locsFile",
                                       llvm::cl::desc("List of locations"),
                                       llvm::cl::value_desc("filepath"),
                                       llvm::cl::cat(MyToolCategory));

int main(int argc, const char **argv) {
  using namespace clang::tooling;

  // CommonOptionsParser constructor will parse arguments and create a
  // CompilationDatabase.  In case of error it will terminate the program.
  auto OptionsParser = CommonOptionsParser::create(argc, argv, MyToolCategory);

  if (!OptionsParser) {
    llvm::errs() << OptionsParser.takeError() << "\n";
    return 1;
  }

  ClangTool Tool(OptionsParser->getCompilations(),
                 OptionsParser->getSourcePathList());

  clang::ast_matchers::MatchFinder Finder;

  std::vector<std::string> sourceLocs;
  {
    std::ifstream stream{locs};
    while (stream) {
      std::string s;
      stream >> s;
      if (s.size())
        sourceLocs.emplace_back(s);
    }

    for (auto &elem : sourceLocs) {
      llvm::outs() << elem << "\n";
    }
  }
  nus::test::StmtCollector Collector(sourceLocs);

  {
    using namespace clang::ast_matchers;
    Finder.addMatcher(stmt(unless(hasParent(functionDecl()))).bind("stmt"),
                      &Collector);
  }
  return Tool.run(newFrontendActionFactory(&Finder).get());
}