#include <golly/Analysis/PscopBuilder.h>
#include <golly/Analysis/RaceDetection.h>

namespace golly {
AnalysisKey RaceDetector::Key;

Races RaceDetector::run(Function &f, FunctionAnalysisManager &fam) {
  auto pscop = fam.getResult<golly::PscopBuilderPass>(f);

  return {};
}
} // namespace golly