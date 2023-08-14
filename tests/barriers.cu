__global__ void barriers(int *arr, int N) {
  if (threadIdx.x < 32) {
    arr[threadIdx.x] = 1;
    __syncwarp(0b0011); // 0,1 , 4, 5
    arr[threadIdx.x + 1] = 2;
    __syncwarp();
  }

  __syncthreads();
}