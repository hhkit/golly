__global__ void ballot() {
  auto mask = __ballot_sync(-1, threadIdx.x > 32);
  __syncwarp(mask);
}