__global__ void yolo(int *arr) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx == 0) {
    arr[1] = 5;
  }

  arr[idx] = threadIdx.x;
}