__global__ void warp_sync() {
  if (threadIdx.x < 16)
    __syncwarp();
}