#include <cassert>

__global__ void test(int val) { assert(0 < val); }