#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <iomanip>

#include "RigidBody.cuh"
#include "../GpuErrorHandling.hpp"
#include "../Constants.hpp"

__global__ void calculatMassCenterKern(int n, int* points, float* x, float* y, float* z, float* invmass, float* cx, float* cy, float* cz, float* tm)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n)	return;
	

	atomicAdd(cx, x[points[i]] / invmass[points[i]]);
	atomicAdd(cy, y[points[i]] / invmass[points[i]]);
	atomicAdd(cz, z[points[i]] / invmass[points[i]]);

	atomicAdd(tm, 1.0f/invmass[points[i]]);
}

__global__ void calculateRadiusKern(int n, int* points, float* x, float* y, float* z, float* invmass, float* cx, float* cy, float* cz, float* rx, float* ry, float* rz)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n)	return;

	rx[i] = x[points[i]] - *cx;
	ry[i] = y[points[i]] - *cy;
	rz[i] = z[points[i]] - *cz;
}

__global__ void calculateAKern(int n, int* points, float* A, float* rx, float* ry, float* rz, float* invmass)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n)	return;

	int index = points[i];

	atomicAdd(&A[0], invmass[index] * (rx[i] * rz[i]));
	atomicAdd(&A[1], invmass[index] * (rx[i] * ry[i]));
	atomicAdd(&A[2], invmass[index] * (rx[i] * rz[i]));
	atomicAdd(&A[3], invmass[index] * (rx[i] * ry[i]));
	atomicAdd(&A[4], invmass[index] * (ry[i] * ry[i]));
	atomicAdd(&A[5], invmass[index] * (ry[i] * rz[i]));
	atomicAdd(&A[6], invmass[index] * (rx[i] * rz[i]));
	atomicAdd(&A[7], invmass[index] * (ry[i] * rz[i]));
	atomicAdd(&A[8], invmass[index] * (rz[i] * rz[i]));
}

RigidBody::RigidBody(std::vector<int> points, float* x, float* y, float* z, float* invmass)
{
	n = points.size();
	gpuErrchk(cudaMalloc((void**)&(this->points), n * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(this->rx), n * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&(this->ry), n * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&(this->rz), n * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&(this->A), matrixSize * sizeof(float)));

	gpuErrchk(cudaMemcpy(this->points, points.data(), n * sizeof(int), cudaMemcpyHostToDevice));

	calculateRadius(x, y, z, invmass);
	calculateA(invmass);
}

RigidBody::~RigidBody()
{
	gpuErrchk(cudaFree(points));

	gpuErrchk(cudaFree(rx));
	gpuErrchk(cudaFree(ry));
	gpuErrchk(cudaFree(rz));
}

void RigidBody::calculateRadius(float* x, float* y, float* z, float* invmass)
{
	int threads = 32;
	int blocks = (n + threads - 1) / threads;
	float* cx, *cy, *cz, *tm;
	gpuErrchk(cudaMalloc((void**)&cx, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&cy, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&cz, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&tm, sizeof(float)));

	calculatMassCenterKern<< <blocks, threads >> >(n, points, x, y, z, invmass, cx, cy, cz, tm);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::device_ptr<float> cx_ptr(cx);
	thrust::device_ptr<float> cy_ptr(cy);
	thrust::device_ptr<float> cz_ptr(cz);
	thrust::device_ptr<float> tm_ptr(tm);

	*cx_ptr /= *tm_ptr;
	*cy_ptr /= *tm_ptr;
	*cz_ptr /= *tm_ptr;

	calculateRadiusKern<< <blocks, threads >> >(n, points, x, y, z, invmass, cx, cy, cz, rx, ry, rz);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void RigidBody::calculateA(float* invmass)
{
	int threads = 32;
	int blocks = (n + threads - 1) / threads;

	calculateAKern << <blocks, threads >> > (n, points, A, rx, ry, rz, invmass);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::device_ptr<float> A_ptr{ A };
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			std::cout << std::setw(3) << A_ptr[3 * i + j] <<  " ";
		std::cout << std::endl;
	}
}
