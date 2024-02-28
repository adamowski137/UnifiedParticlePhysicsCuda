#include "FloorConstrain.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "../../GpuErrorHandling.hpp"


FloorConstrain::FloorConstrain(int* indexes) : Constrain{ 1, 1.0f, -10000.0f, 10000.0f, indexes }
{
}

void FloorConstrain::fillJacobian(float* jacobianRow)
{
	int threads = 512;
	int blocks = ceilf((float)n / threads);
	fillJacobianKern<<<threads, blocks>>>(n, jacobianRow, dev_indexes);
}

__global__ void fillJacobianKern(int n, float* jacobianRow, int* idx)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) return;
	jacobianRow[idx[index]] = 1.0f;
}






