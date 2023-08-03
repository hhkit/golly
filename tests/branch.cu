__global__ void branch(int *val) {
  if (threadIdx.x > 2)
    val[0] = 1;
  else
    val[1] = 2;

  if (threadIdx.x != 0) {
    val[2] = 7;
  }

  for (int i = 0; i < 10; ++i) {
    if (i > 2)
      val[i] = 4;
    else
      val[i + 1] = 5;
  }
}