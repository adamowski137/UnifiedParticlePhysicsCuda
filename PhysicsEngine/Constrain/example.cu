// example.cuh

#ifndef EXAMPLE_CUH
#define EXAMPLE_CUH

#include <cuda_runtime.h>

__global__ void kernelFunction(int* array, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        array[tid] = tid * tid;
    }
}

void launchKernel(int* array, int size) {
    int* d_array;
    cudaMalloc(&d_array, size * sizeof(int));

    kernelFunction < < <1, size > > > (d_array, size);

    cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_array);
}

#endif