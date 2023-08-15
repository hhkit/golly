__global__ void yolo(int *val) {
  if (threadIdx.x < 16)
    val[threadIdx.x + 1] = 0;
  else
    val[threadIdx.x] = 1;
  __syncthreads();
}