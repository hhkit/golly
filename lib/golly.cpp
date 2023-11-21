#include "golly/golly.h"
#include "golly/Analysis/ConditionalDominanceAnalysis.h"
#include "golly/Analysis/CudaParameterDetection.h"
#include "golly/Analysis/PolyhedralBuilder.h"
#include "golly/Analysis/PscopDetector.h"
#include "golly/Analysis/RaceDetection.h"
#include "golly/Analysis/SccOrdering.h"
#include "golly/Analysis/StatementDetection.h"
#include "golly/ErrorHandling/YamlDumper.h"
#include "golly/Support/GollyOptions.h"

#include <llvm/Analysis/RegionInfo.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Transforms/Scalar/LoopRotation.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>

namespace golly {
static std::string symbolOnly(std::string_view str) {
  auto pos = str.find_first_of("(");
  return std::string(str.substr(0, pos));
}
namespace detail {

static bool checkParametrizedPassName(llvm::StringRef Name,
                                      llvm::StringRef PassName) {
  if (!Name.consume_front(PassName))
    return false;
  // normal pass name w/o parameters == default parameters
  if (Name.empty())
    return true;
  return Name.startswith("<") && Name.endswith(">");
}

// stole this off the llvm source code
template <typename ParametersParseCallableT>
auto parsePassParameters(ParametersParseCallableT &&Parser,
                         llvm::StringRef Name, llvm::StringRef PassName)
    -> decltype(Parser(llvm::StringRef{})) {
  using ParametersT = typename decltype(Parser(llvm::StringRef{}))::value_type;

  llvm::StringRef Params = Name;
  if (!Params.consume_front(PassName)) {
    assert(false &&
           "unable to strip pass name from parametrized pass specification");
  }
  if (!Params.empty() &&
      (!Params.consume_front("<") || !Params.consume_back(">"))) {
    assert(false && "invalid format for parametrized pass name");
  }

  llvm::Expected<ParametersT> Result = Parser(Params);
  assert((Result || Result.template errorIsA<llvm::StringError>()) &&
         "Pass parameter parser can only return StringErrors.");
  return Result;
}

} // namespace detail

static llvm::Optional<GollyOptions> options;

llvm::Optional<GollyOptions> RunGollyPass::getOptions() { return options; }

golly::RunGollyPass::RunGollyPass(GollyOptions &opt) { options = opt; }

PreservedAnalyses RunGollyPass::run(Function &f, FunctionAnalysisManager &fam) {
  if (f.getName() == "_Z10__syncwarpj") {
    return PreservedAnalyses::none();
  }

  auto params = fam.getResult<golly::CudaParameterDetection>(f);

  llvm::outs() << "Race detection of ";
  llvm::WithColor(llvm::outs(), llvm::HighlightColor::Address)
      << symbolOnly(llvm::demangle(f.getName().str()));
  llvm::outs() << " using " << params << ": ";

  if (auto errs = fam.getResult<golly::RaceDetector>(f)) {
    if (options->errorLog)
      dumpYaml(errs, *options->errorLog);

    llvm::outs() << "\n";
    for (auto &elem : errs) {
      llvm::WithColor(llvm::outs(), llvm::HighlightColor::Error) << "ERROR: ";
      elem.print(llvm::WithColor(llvm::outs(), llvm::HighlightColor::Warning));
      llvm::outs() << " detected "
                      "\n";
    }

  } else
    llvm::WithColor(llvm::outs(), llvm::raw_ostream::GREEN, true) << "Clear!\n";

  return PreservedAnalyses::all();
}
} // namespace golly

llvm::PassPluginLibraryInfo getGollyPluginInfo() {
  using llvm::ArrayRef;
  using llvm::PassBuilder;
  using llvm::StringRef;

  return {
      LLVM_PLUGIN_API_VERSION, "golly", LLVM_VERSION_STRING,
      [](PassBuilder &PB) {
        PB.registerAnalysisRegistrationCallback(
            [](llvm::FunctionAnalysisManager &fam) {
              fam.registerPass([]() { return golly::PolyhedralBuilderPass(); });
              fam.registerPass(
                  []() { return golly::ConditionalDominanceAnalysisPass(); });
              fam.registerPass([]() { return golly::SccOrderingAnalysis(); });
              fam.registerPass(
                  []() { return golly::CudaParameterDetection(); });
              fam.registerPass(
                  []() { return golly::StatementDetectionPass(); });
              fam.registerPass([]() { return golly::PscopDetectionPass(); });
              fam.registerPass([]() { return golly::RaceDetector(); });
            });

        PB.registerPipelineParsingCallback(
            [](StringRef Name, llvm::FunctionPassManager &PM,
               ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
              // if (Name == "golly") {
              //   // PM.addPass(golly::RunGollyPass());
              //   // return true;
              // }
              if (golly::detail::checkParametrizedPassName(Name, "golly")) {
                auto Params = golly::detail::parsePassParameters(
                    golly::parseOptions, Name, "golly");
                if (!Params) {
                  return false;
                }
                PM.addPass(golly::RunGollyPass(Params.get()));
                return true;
              }

              return false;
            });
      }};
}

#ifndef LLVM_GOLLY_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getGollyPluginInfo();
}
#endif
