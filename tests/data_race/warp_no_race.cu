__global__ void warp_no_race(int *val) {
  if (blockIdx.x == 0)
    if (threadIdx.x < 32) {
      auto v = val[threadIdx.x] + val[threadIdx.x + 32];
      __syncwarp();
      val[threadIdx.x] = v;
      __syncwarp();
      v = val[threadIdx.x] + val[threadIdx.x + 16];
      __syncwarp();
      val[threadIdx.x] = v;
      __syncwarp();
    }
}