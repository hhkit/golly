__global__ void simple(int *val) {
  int sum = 0;
  for (int i = 0; i < 16; ++i)
    sum += i;
  val[threadIdx.x] = 0;
  __syncwarp();
}