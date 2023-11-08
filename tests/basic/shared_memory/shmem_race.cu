
// extern int p;

__global__ void test(int *a) {
  __shared__ int arr[256];

  arr[1] = 0;
}
