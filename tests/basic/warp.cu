__global__ void warp(int *arr) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    __syncwarp(0b101);
}