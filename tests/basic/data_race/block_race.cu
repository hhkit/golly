__global__ void trivial_race(int *val, int *val2) {
  __shared__ int shmem[256];
  shmem[0] = 1;
  __syncthreads();
}