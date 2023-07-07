__global__ void loop(int* arr, int N){
    for (int i = 0; i < N; ++i) {
        arr[threadIdx.x] += i * N;
    }

}