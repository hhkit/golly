__global__ void memcpy(int *dst, int *src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  dst[i] = src[i];
}