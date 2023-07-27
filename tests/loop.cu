__global__ void simple(int *val, int N) {
  for (int i = 0; i < N; ++i)
    val[i]++;
  __syncwarp();
}