__global__ void yolo(int *arr) {
  extern __shared__ int mem[];
  auto myid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = 0; i < 3; ++i) {
    if (myid < 16) {
      mem[myid + 1] = 0;
    } else {
      mem[myid] = 0;
    }
    __syncthreads();
  }
}