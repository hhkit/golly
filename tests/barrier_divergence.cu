__global__ void block_div() {
  if (threadIdx.x < 64)
    __syncthreads();
}