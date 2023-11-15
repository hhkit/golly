#include <clang/Frontend/FrontendActions.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/CommandLine.h>

#include <clang/ASTMatchers/ASTMatchFinder.h>        // MatchFinder
#include <clang/Tooling/Transformer/RangeSelector.h> // RangeSelector
#include <clang/Tooling/Transformer/Stencil.h>       // cat

#include "MatchPrinter.h"
#include "RewritePrinter.h"

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("my-tool options");
static llvm::cl::extrahelp
    CommonHelp(clang::tooling::CommonOptionsParser::HelpMessage);

int main(int argc, const char **argv) {
  using namespace clang::tooling;
  using namespace clang::ast_matchers;

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
  nus::test::MatchPrinter Printer;
  auto MatchRule =
      functionDecl(hasAttr(clang::attr::Kind::CUDAGlobal)).bind("func");

  Finder.addMatcher(MatchRule, &Printer);

  return Tool.run(newFrontendActionFactory(&Finder).get());
}
