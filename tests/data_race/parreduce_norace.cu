__global__ void parreduce() {
  extern __shared__ int shmem[];
  unsigned tid = threadIdx.x;

  if (blockIdx.x == 0 && threadIdx.x < 32) {
    // Incorrect use of __syncwarp()
    auto v = shmem[tid] + shmem[tid + 16];
    __syncwarp();
    shmem[tid] = v;
    __syncwarp();
    v = shmem[tid] + shmem[tid + 8];
    __syncwarp();
    shmem[tid] = v;
    __syncwarp();
    v = shmem[tid] + shmem[tid + 4];
    __syncwarp();
    shmem[tid] = v;
    __syncwarp();
    v = shmem[tid] + shmem[tid + 2];
    __syncwarp();
    shmem[tid] = v;
    __syncwarp();
    v = shmem[tid] + shmem[tid + 1];
    __syncwarp();
    shmem[tid] = v;
    __syncwarp();
  }
}