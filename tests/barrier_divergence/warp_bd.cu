__global__ void block_div() {
  if (threadIdx.x % 2 == 1)
    __syncwarp(0x77777777);
}