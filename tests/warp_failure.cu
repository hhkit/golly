__global__ void warp_sync() {
  if (threadIdx.x < 32)
    __syncwarp();
}