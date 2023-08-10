__global__ void block_div() {
  if (threadIdx.x > 255)
    __syncthreads();
}