#include <cassert>

__global__ void yolo(int *arr) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  assert(idx > 5);
  if (!(idx > 5))
    return;

  if (idx == 0) {
    arr[1] = 5;
  } else {
    arr[0] = 1;
  }

  arr[idx] = threadIdx.x;
}