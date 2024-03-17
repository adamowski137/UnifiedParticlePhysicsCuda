#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "LinearSolver.cuh"
#include <cmath>
#include <chrono>
#include <iostream>
#include "../GpuErrorHandling.hpp"

__global__ void jaccobiKern(int n, float* A, float* b, float* x, float* outX, float* c_min, float* c_max)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) return;
	float a = A[index * n + index];
	float cx = b[index];
	for (int i = 0; i < n; i++)
	{
		if (i == index) continue;
		cx -= (A[index * n + i] * x[index]);
	}
	//float out = cx / a;
	//out = out > c_min[index] ? out : c_min[index];
	//out = out < c_max[index] ? out : c_max[index];
	outX[index] = min(max(cx / a, c_min[index]), c_max[index]);
	//outX[index] = out;
//	outX[index] = cx / a;
	

}

void jaccobi(int n, float* A, float* b, float* x, float* new_x, float* c_min, float* c_max, int iterations)
{
	int threadsPerBlock = 512;
	int blocks = ceilf((float)n / threadsPerBlock);
	for (int i = 0; i < iterations; i++)
	{
		jaccobiKern << <blocks, threadsPerBlock >> > (n, A, b, x, new_x, c_min, c_max);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());

		std::swap(new_x, x);
	}
}