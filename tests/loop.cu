__global__ void loop(int *arr, int N) {
  if (blockIdx.x < 2) {
    arr[0] = gridDim.x;
  }

  for (int i = 0; i < blockDim.x; ++i) {
    arr[threadIdx.x] += i * N;
  }

  for (int i = 0; i < blockDim.y; ++i) {
    arr[threadIdx.y] += i * N;
  }
}