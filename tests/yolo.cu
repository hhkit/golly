__global__ void yolo(int *val) {
  for (int i = 0; i < 3; ++i) {
    if (threadIdx.x < 16)
      val[threadIdx.x + 1] = 0;
    else
      val[threadIdx.x] = 1;
    __syncthreads();
  }
}