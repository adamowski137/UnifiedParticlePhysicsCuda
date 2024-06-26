#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "LinearSolver.cuh"
#include <cmath>
#include <chrono>
#include <iostream>
#include "../GpuErrorHandling.hpp"

__global__ void jaccobiKern(int n, int nIterations, float* A, float* b, float* x, float* outX, float* c_min, float* c_max)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) return;
	for (int k = 0; k < nIterations; k++)
	{
		float a = A[index * n + index];
		float cx = b[index];
		for (int i = 0; i < n; i++)
		{
			if (i == index) continue;
			cx -= (A[index * n + i] * x[index]);
		}
		outX[index] = min(max(cx / a, c_min[index]), c_max[index]);

		x[index] = outX[index];
		__syncthreads();
	}
}

void jaccobi(int n, float* A, float* b, float* x, float* new_x, float* c_min, float* c_max, int iterations)
{
	int threadsPerBlock = 512;
	int blocks = ceilf((float)n / threadsPerBlock);
	jaccobiKern << <blocks, threadsPerBlock >> > (n, iterations, A, b, x, new_x, c_min, c_max);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void gauss_seidel_cpu(int n, float* A, float* b, float* x, float* new_x, float* c_min, float* c_max, int iterations)
{
	float* cpu_A = new float[n * n];
	float* cpu_b = new float[n];
	float* cpu_x = new float[n];
	float* cpu_new_x = new float[n];
	float* cpu_cmin = new float[n];
	float* cpu_cmax = new float[n];

	cudaMemcpy(cpu_A, A, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_b, b, sizeof(float) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_x, x, sizeof(float) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_new_x, new_x, sizeof(float) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_cmin, c_min, sizeof(float) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_cmax, c_max, sizeof(float) * n, cudaMemcpyDeviceToHost);


	for (int k = 0; k < iterations; k++)
	{
		for (int i = 0; i < n; i++)
		{
			float delta = 0.0f;
			for (int j = 0; j < n; j++)
			{
				if (i == j) continue;
				delta += cpu_A[i * n + j] * cpu_new_x[j];
			}
			cpu_new_x[i] = (cpu_b[i] - delta) / cpu_A[i * n + i];

			// projected gauss seidel
			cpu_new_x[i] = std::min(std::max(cpu_new_x[i], cpu_cmin[i]), cpu_cmax[i]);
		}
	}



	cudaMemcpy(A, cpu_A, sizeof(float) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(b, cpu_b, sizeof(float) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(x, cpu_x, sizeof(float) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(new_x, cpu_new_x, sizeof(float) * n, cudaMemcpyHostToDevice);


	delete[] cpu_A, cpu_b, cpu_x, cpu_new_x, cpu_cmin, cpu_cmax;
}
