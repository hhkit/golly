#include "golly/Support/GollyOptions.h"

#include <fstream>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <ryml.hpp>
#include <sstream>
#include <string>

namespace golly {

static std::string trim(const std::string &str) {
  size_t first = str.find_first_not_of(' ');
  if (std::string::npos == first) {
    return str;
  }
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}

llvm::Expected<GollyOptions> parseOptions(llvm::StringRef params) {
  if (params.empty())
    return GollyOptions();

  GollyOptions options;
  while (!params.empty()) {
    if (params.consume_front("outfile=")) {
      auto token = params.take_while(
          [](char c) -> bool { return c != EOF && c != ';'; });
      params.consume_front(token);
      if (params.consume_front(";") || params.empty())
        continue;
      options.errorLog = token.str();
      continue;
    }

    if (params.consume_front("config=")) {
      auto token = params.take_while(
          [](char c) -> bool { return c != EOF && c != ';'; });
      params.consume_front(token);
      if (params.consume_front(";") || params.empty()) {
        if (auto buf = llvm::MemoryBuffer::getFile(token, true)) {
          auto config = ryml::parse_in_arena((*buf)->getBuffer().data());
          assert(config.is_seq(0) && "for now, only accept arrays");
          auto root = config.rootref();

          for (auto &&elem : root.children()) {
            assert(elem["name"].has_val());
            std::stringstream sstream;
            sstream << elem["name"].val();
            auto func_name = trim(sstream.str());

            auto parse_dim3 = [](c4::yml::NodeRef ref) -> dim3 {
              dim3 ret;
              if (ref.is_seq()) {
                auto ptr = &ret.x;
                for (auto &&var : ref.children())
                  *ptr++ = std::stoi(var.val().data());
              } else {
                assert(ref.is_keyval());
                if (ref.is_keyval())
                  ret.x = std::stoi(ref.val().data());
              }
              return ret;
            };

            options.function_parameters[func_name].block =
                parse_dim3(elem["block"]);
            options.function_parameters[func_name].grid =
                parse_dim3(elem["grid"]);
          }
        }
        continue;
      }
    }

    if (params.consume_front("verbose")) {
      if (params.consume_front(";") || params.empty()) {
        options.verboseLog = true;
        continue;
      }
    }
    return llvm::make_error<llvm::StringError>(
        llvm::formatv("not implemented").str(), llvm::inconvertibleErrorCode());
  }
  return options;
}
} // namespace golly