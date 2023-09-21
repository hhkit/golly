__global__ void yolo(int *arr) {
  for (int i = 0; i < 5; ++i)
    // for (int j = threadIdx.x; j < 3 * threadIdx.x + threadIdx.x; ++j)
    arr[0] = 7;
}