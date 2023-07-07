__global__ void simple(int* val){
        val[threadIdx.x] = 0;
}