#ifndef GOLLY_ERRORHANDLING_YAMLDUMPER_H
#define GOLLY_ERRORHANDLING_YAMLDUMPER_H
#include "Error.h"
#include <filesystem>

namespace golly {
void dumpYaml(const ErrorList &errs, std::filesystem::path yaml_out);
}

#endif /* GOLLY_ERRORHANDLING_YAMLDUMPER_H */
