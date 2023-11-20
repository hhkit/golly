#include "golly/ErrorHandling/YamlDumper.h"
#include "golly/Analysis/Statements.h"

#include <fstream>
#include <iostream>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/DebugLoc.h>
#include <ryml.hpp>
#include <ryml_std.hpp>

namespace golly {
static const llvm::DILocation &getLoc(const Statement *stmt) {
  return *stmt->getDefiningInstruction().getDebugLoc().get();
};
void dumpYaml(const ErrorList &errs, std::filesystem::path yaml_out) {
  namespace fs = std::filesystem;

  ryml::Tree tree;
  auto root = tree.rootref();
  root |= ryml::SEQ;

  for (auto &err : errs) {
    if (auto as_race = std::get_if<golly::DataRace>(&err)) {
      auto err = root.append_child();
      err |= ryml::MAP;
      err["type"] = "race";

      auto level = err["level"];
      switch (as_race->level) {
      case Level::Grid:
        level << "grid";
        break;
      case Level::Block:
        level << "block";
        break;
      case Level::Warp:
        level << "warp";
        break;
      }

      auto locs = err["locs"];
      locs |= ryml::SEQ;

      auto &write_dbg_loc = getLoc(as_race->instr1);
      auto &acc_dbg_loc = getLoc(as_race->instr2);

      auto add_loc = [](ryml::NodeRef store, const llvm::DILocation &dbg) {
        constexpr auto canonicalize =
            [](const llvm::DILocation &dbg) -> fs::path {
          return fs::canonical(fs::path{dbg.getDirectory().str()} /
                               fs::path{dbg.getFilename().str()});
        };

        auto path = canonicalize(dbg).string();
        auto line = dbg.getLine();
        auto col = dbg.getColumn();

        store |= ryml::MAP;
        store["path"] << path;
        store["line"] << line;
        store["col"] << col;
      };

      add_loc(locs.append_child(), write_dbg_loc);
      add_loc(locs.append_child(), acc_dbg_loc);
    }

    if (auto bd = std::get_if<golly::BarrierDivergence>(&err)) {
      auto err = root.append_child();
      err |= ryml::MAP;
      err["type"] = "bd";

      auto level = err["level"];
      switch (bd->level) {
      case Level::Grid:
        level << "grid";
        break;
      case Level::Block:
        level << "block";
        break;
      case Level::Warp:
        level << "warp";
        break;
      }

      // auto loc = err["locs"];
      // loc |= ryml::SEQ;

      // auto &write_dbg_loc = getLoc(bd-);
      // auto add_loc = [](ryml::NodeRef store, const llvm::DILocation &dbg) {
      //   constexpr auto canonicalize =
      //       [](const llvm::DILocation &dbg) -> fs::path {
      //     return fs::canonical(fs::path{dbg.getDirectory().str()} /
      //                          fs::path{dbg.getFilename().str()});
      //   };

      //   auto path = canonicalize(dbg).string();
      //   auto line = dbg.getLine();
      //   auto col = dbg.getColumn();

      //   store |= ryml::MAP;
      //   store["path"] << path;
      //   store["line"] << line;
      //   store["col"] << col;
      // };
    }
  }
  // llvm::dbgs() << std::string{"writing"} << "\n";
  if (auto out = std::ofstream{yaml_out}) {
    out << tree;
    // llvm::dbgs() << "wrote to " << yaml_out << "\n";
  }
}
} // namespace golly