__global__ void loop(int *arr, int N) {
  if (blockIdx.x < 2) {
    arr[0] = gridDim.x;
  }

  for (int i = 0; i < blockDim.x; ++i) {
    arr[threadIdx.x] += i * N;
    arr[threadIdx.x] *= 2;
  }
  __syncwarp();

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < i; ++j)
      arr[threadIdx.y] += j;
  }

  for (int i = 16; i > 0; i /= 2) {
    __syncwarp();
  }
}