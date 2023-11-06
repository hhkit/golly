__global__ void branch(int *val) {
  if (threadIdx.x > 2)
    val[0] = val[1];
  else
    val[1] = 2;
}