__global__ void trivial_race(int *val, int *val2) {
  *val = threadIdx.x;
  val2[threadIdx.x] = threadIdx.x;
}