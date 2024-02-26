#include "FloorConstrain.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "../../GpuErrorHandling.hpp"


FloorConstrain::FloorConstrain()
{
}

void FloorConstrain::fillJacobian(int particles, int constrains, float* jacobian)
{
	int threads = 512;
	int blocks = ceilf((float)constrains/threads);
	fillJacobianKern<<<threads, blocks>>>(particles, 3 * particles * constrains, jacobian);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

}

__global__ void fillJacobianKern(int particles, int size, float* jacobian)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= size) return;
	int row = index / particles;
	jacobian[row * 3 + 1] = 1;
}






