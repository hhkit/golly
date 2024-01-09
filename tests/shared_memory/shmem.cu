
// extern int p;

__global__ void test(int *a) {
  __shared__ int arr[256];
  extern __shared__ int p[];

  arr[threadIdx.x] = 0;
  p[threadIdx.x] = 1;
  // a[threadIdx.x] = 2;
}

// __global__ void test2() {
//   __shared__ int arr[256];
//   extern __shared__ int p[];

//   arr[threadIdx.x] = 0;
//   p[threadIdx.x] = 1;
// }