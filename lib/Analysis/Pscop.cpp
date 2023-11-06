#include "golly/Analysis/Pscop.h"
#include "golly/Support/isl_llvm.h"

namespace golly {
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Pscop &pscop) {
  os << "domain:\n  " << pscop.instantiation_domain << "\n";
  os << "thread_allocation:\n  " << pscop.thread_allocation << "\n";
  os << "temporal_schedule:\n  " << pscop.temporal_schedule << "\n";
  os << "sync_schedule:\n  " << pscop.sync_schedule << "\n";
  os << "writes:\n  " << pscop.write_access_relation << "\n";
  os << "reads:\n  " << pscop.read_access_relation << "\n";
  return os;
}
} // namespace golly