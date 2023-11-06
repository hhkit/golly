#ifndef GOLLY_ANALYSIS_PSCOP_H
#define GOLLY_ANALYSIS_PSCOP_H

#include "golly/Analysis/Error.h"
#include "golly/Support/isl.h"
#include <llvm/Support/raw_ostream.h>

namespace golly {
/*

Synchronization can only be determined between two threads. A single thread is
insufficient for determining when it synchronizes with other threads.

a note on the type annotations:
  Stmt:
    A statement - an atomic subdivision of instructions that we care about

  StmtInst:
    An instance of a statement, duplicated by parallelism or loops.
    eg. StmtA[0, 1] \in { StmtA[tid, i] : 0 <= tid <= 15 and 0 <= i <= 2}

  tid:
    Thread ID

  Time:
    A multidimensional vector, the lexicographic ordering of which corresponds
    to temporal dependence within a thread

  Access:
    A single dimensional vector that represents an access of a memory location
*/
struct Pscop {
  islpp::union_map instantiation_domain; // { Stmt -> StmtInst }
  islpp::union_map thread_allocation;    // { StmtInst -> tid }
  islpp::union_map temporal_schedule;    // { StmtInst -> Time }

  islpp::union_map sync_schedule; // { { tid -> tid } -> StmtInst }

  islpp::union_map write_access_relation; // param -> { StmtInst -> Access }
  islpp::union_map read_access_relation;  // param -> { StmtInst -> Access }

  // dependence relation irrelevant for race detection
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Pscop &);
} // namespace golly

#endif /* GOLLY_ANALYSIS_PSCOP_H */
