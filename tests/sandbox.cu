__global__ void yolo(int *val) {
  extern __shared__ int shared_mem[];
  auto myid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = 0; i < 3; ++i) {
    if (myid < 16) {
      shared_mem[myid + 1] = 0;
    } else {
      // auto j = val[threadIdx.x];
      shared_mem[myid] = 0;
    }
    __syncthreads();
  }
}