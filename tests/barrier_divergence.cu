__global__ void block_div() {
  if (threadIdx.x < 128)
    __syncthreads();
}