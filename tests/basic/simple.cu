__global__ void simple(int *val, int N) { val[threadIdx.x] = 0; }
