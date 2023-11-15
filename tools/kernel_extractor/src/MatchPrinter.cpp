#include "MatchPrinter.h"
#include <clang/Lex/Lexer.h>
#include <fstream>
namespace nus::test {
namespace detail {
std::string decl2str(const clang::SourceRange &sr,
                     const clang::ASTContext &ac) {
  auto &sm = ac.getSourceManager();
  auto &lo = ac.getLangOpts();
  // (T, U) => "T,,"
  auto text = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(sr), sm, lo);
  if (text.size() > 0 && (text.back() == ',')) // the text can be
                                               // ""
    return clang::Lexer::getSourceText(clang::CharSourceRange::getCharRange(sr),
                                       sm, lo, 0)
        .str();
  return text.str();
}

} // namespace detail
void MatchPrinter::run(const MatchResult &Result) {
  for (auto [id, node] : Result.Nodes.getMap()) {
    {
      if (auto as_decl = node.get<clang::FunctionDecl>()) {
        llvm::outs() << *as_decl << "\n";
        auto &sm = *Result.SourceManager;
        auto loc =
            sm.getSpellingLoc(as_decl->getBody()->getSourceRange().getBegin());
        auto filepath =
            Result.SourceManager->getFilename(loc).str() + ".ext.cu";
        llvm::outs() << "extracting from "
                     << loc.printToString(*Result.SourceManager) << " to "
                     << filepath << "\n";
        if (auto open_file = std::ofstream{filepath}) {
          open_file << detail::decl2str(node.getSourceRange(), *Result.Context);
          llvm::outs() << "wrote to " << filepath << "\n";
        } else {
          llvm::outs() << "NO\n";
        }
      }
    }
  }
}
} // namespace nus::test