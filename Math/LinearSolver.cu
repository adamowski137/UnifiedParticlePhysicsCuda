#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "LinearSolver.cuh"
#include <cmath>
#include <chrono>
#include <iostream>

#define EPS 0.00001

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void jaccobiKern(int n, float* A, float* b, float* x, float* outX)
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
	outX[index] = cx / a;
}

__global__ void gausSeidlKern(int n, float* A, float* b, float* x, float* outX)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) return;

}

__global__ void colorGraphKern(int n, float* A, float* colors, float* minColor, bool* U)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n || colors[index] != -1) return;
	if (U[index])
	{
		for (int i = 0; i < n; i++)
		{
			if (index == i) continue;
			if (A[index * n + i] < EPS) continue;
			minColor[i] = max(minColor[i], colors[index]);
		}
		colors[index] = minColor[index] + 1;
	}

}
__global__ void detectConflictsKern(int n, float* A, float* colors, bool* U, bool* R, int* amountOfConflicts)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n || colors[index] != -1) return;
	if (U[index])
	{
		for (int i = 0; i < n; i++)
		{
			if (index == i) continue;
			if (A[index * n + i] < EPS) continue;
			if (colors[i] == colors[index])
			{
				R[max(i, index)] = true;
				*amountOfConflicts += 1;
			}
		}
	}
}



void jaccobi(int n, float* A, float* b, float* x)
{
	float* dev_a, * dev_b, * dev_x, * dev_nx;
	cudaMalloc((void**)&dev_a, sizeof(float) * n * n);
	cudaMalloc((void**)&dev_b, sizeof(float) * n);
	cudaMalloc((void**)&dev_x, sizeof(float) * n);
	cudaMalloc((void**)&dev_nx, sizeof(float) * n);

	cudaMemcpy(dev_a, A, sizeof(float) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(float) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);

	int threadsPerBlock = 512;
	int blocks = ceilf((float)n / threadsPerBlock);
	auto start = std::chrono::high_resolution_clock::now();
	jaccobiKern << < threadsPerBlock, blocks >> > (n, dev_a, dev_b, dev_x, dev_nx);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "duration kernel: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	cudaMemcpy(x, dev_nx, sizeof(float) * n, cudaMemcpyDeviceToHost);
}

void gaussSeidl(int n, float* A, float* b, float* x)
{
	float* dev_a, * dev_b, * dev_x, * dev_nx;
	cudaMalloc((void**)&dev_a, sizeof(float) * n * n);
	cudaMalloc((void**)&dev_b, sizeof(float) * n);
	cudaMalloc((void**)&dev_x, sizeof(float) * n);
	cudaMalloc((void**)&dev_nx, sizeof(float) * n);

	cudaMemcpy(dev_a, A, sizeof(float) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(float) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);

	int threadsPerBlock = 512;
	int blocks = ceilf((float)n / threadsPerBlock);
	auto start = std::chrono::high_resolution_clock::now();
	jaccobiKern << < threadsPerBlock, blocks >> > (n, dev_a, dev_b, dev_x, dev_nx);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "duration kernel: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	cudaMemcpy(x, dev_nx, sizeof(float) * n, cudaMemcpyDeviceToHost);
}

void findGraphColoringKern(int n, float* A)
{
	int* colors;

	cudaMalloc((void**)&colors, sizeof(int) * n);
	cudaMemset(colors, -1, sizeof(int) * n);

	int recolor = n;
	while (recolor > 0)
	{

	}

}

