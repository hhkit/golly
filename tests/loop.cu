__global__ void loop(int *val, int N) {
  // for (int i = 0; i < N; ++i)
  //   val[threadIdx.x * 5]++;

  // for (int j = 0; j < 10; ++j)
  //   val[j]++;

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < i; ++j)
      val[j] = 1;
    val[3] = 2;
  }
}