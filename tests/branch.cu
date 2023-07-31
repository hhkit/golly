__global__ void branch(int *val) {
  if (threadIdx.x > 2)
    val[0] = 2;
  else
    val[1] = 1;
}