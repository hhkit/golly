__global__ void yolo(int *val) {
  auto myid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = 0; i < 3; ++i) {
    if (myid < 16) {
      val[myid + 1] = 0;
    } else {
      // auto j = val[threadIdx.x];
      val[myid] = 0;
    }
    __syncthreads();
  }
}