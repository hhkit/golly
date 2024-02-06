__global__ void parreduce(int *array, int *result) {
  __shared__ int tmp[256];

  tmp[threadIdx.x] = array[threadIdx.x];
  __syncthreads();

#pragma clang loop unroll(full)
  for (int i = 1; i < 128; i *= 2) {
    if (threadIdx.x % (2 * i) == 0) {
      tmp[threadIdx.x] += tmp[threadIdx.x + i];
      // __syncthreads(); // barrier divergence if block syncs here
    }
    __syncthreads(); // prevents a race on tmp[threadIdx.x]
  }

  if (threadIdx.x == 0)
    *result = tmp[0]; // write to array 0
}