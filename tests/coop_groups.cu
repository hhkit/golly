#include <cooperative_groups.h>

using namespace cooperative_groups;
__device__ int reduce_sum(thread_group g, int *temp, int val) {
  int lane = g.thread_rank();
  int i = g.size();

  // Each iteration halves the number of active threads
  // Each thread adds its partial sum[i] to sum[lane+i]
  while (i > 0) {
    temp[lane] = val;
    g.sync(); // wait for all threads to store
    if (lane < i)
      val += temp[lane + i];
    g.sync(); // wait for all threads to load

    i /= 2;
  }
  return val; // note: only thread 0 will return full sum
}

using namespace cooperative_groups;
__device__ int reduce_sum(thread_block b, int *temp, int val) {
  int lane = b.thread_rank();
  int i = b.size();

  // Each iteration halves the number of active threads
  // Each thread adds its partial sum[i] to sum[lane+i]
  while (i > 0) {
    temp[lane] = val;
    b.sync(); // wait for all threads to store
    if (lane < i)
      val += temp[lane + i];
    b.sync(); // wait for all threads to load

    i /= 2;
  }
  return val; // note: only thread 0 will return full sum
}