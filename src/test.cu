#include <FMM_tSNE/cuda_hello.h>
#include <cstdio>

__global__ void hello()
{
    printf("hello from GPU thread %d\n", threadIdx.x);
}

void cuda_hello()
{
    hello<<<1, 4>>>();
    cudaDeviceSynchronize();
}
