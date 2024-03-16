#include <cuda_runtime.h>
#include "../List/List.cuh"

__global__ void prefixUpSweep(int* input, int n, int d)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) return;
	input[index + powf(2, d + 1) - 1] += input[index + powf(2, d) - 1]
}

__global__ void prefixDownSweep(int* input, int n, int d)
{

}