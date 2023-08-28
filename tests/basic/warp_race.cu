__global__ void warp_race(int *val) {
  if (threadIdx.x < 32) {
    val[threadIdx.x] = val[threadIdx.x] + val[threadIdx.x + 32];
    __syncwarp();
  }
}