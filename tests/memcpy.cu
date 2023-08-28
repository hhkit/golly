__global__ void memcpy(int *dst, int *src, int n) {
  int i = threadIdx.x;
  if (0 <= i && i < n)
    dst[i] = src[i];
}

// todo : figure out how to make it dst[i] instead of [dst + i] without breaking
// %n