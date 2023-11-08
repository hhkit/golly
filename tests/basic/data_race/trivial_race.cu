__global__ void trivial_race(int *val, int *val2) {
  val[1] = 1;
  __syncthreads();
}