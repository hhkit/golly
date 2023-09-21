__global__ void loop(int *val, int N) {
  // for (int i = 0; i < N; ++i)
  //   val[threadIdx.x * 5]++;

  // for (int j = 0; j < 10; ++j)
  //   val[j]++;

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < i; ++j)
      val[threadIdx.x] = 1;
    // val[3] = 2;
  }

  for (int j = 0; j < 3; ++j)
    val[threadIdx.x] = 5;

  for (int j = threadIdx.x; j < 3 * threadIdx.x + threadIdx.x; ++j)
    val[threadIdx.x] = 7;
}