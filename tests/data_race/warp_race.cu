__global__ void warp_race(int *val) {
  if (blockIdx.x == 0)
    if (threadIdx.x < 32) {
      val[threadIdx.x] = val[threadIdx.x] + val[threadIdx.x + 32];
      __syncwarp();
      val[threadIdx.x] = val[threadIdx.x] + val[threadIdx.x + 16];
      __syncwarp();
    }
}