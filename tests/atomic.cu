__global__ void atomic(int *mem) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid != 0)
    atomicAdd(mem, 2);
  else
    *mem = 3;

  atomicAdd_block(mem, 4);
  atomicAdd_system(mem, 5);
}