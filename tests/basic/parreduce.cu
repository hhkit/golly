__global__ void parreduce() {
  extern __shared__ int shmem[];
  unsigned tid = threadIdx.x;

  // Incorrect use of __syncwarp()
  shmem[tid] += shmem[tid + 16];
  __syncwarp();
  shmem[tid] += shmem[tid + 8];
  __syncwarp();
  shmem[tid] += shmem[tid + 4];
  __syncwarp();
  shmem[tid] += shmem[tid + 2];
  __syncwarp();
  shmem[tid] += shmem[tid + 1];
  __syncwarp();
}