// fn_inline.cu
// only analyze __global__ functions
// and
// inline all __device__ functions.

__device__ void inlineme(int *arr, int i) { arr[i] = 2; }

__global__ void fn_test(int *arr) {
  //   int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = blockIdx.x;
  inlineme(arr, tid);
}