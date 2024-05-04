#include "RigidBodyConstraint.cuh"
#include "../../Math/polar/polar_decomposition_3x3.h"
#include "../../GpuErrorHandling.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

__global__ void calculateShapeCovarianceKern(float* A, float* x, float* y, float* z, int* p, int n, float* rx, float* ry, float* rz, float cx, float cy, float cz)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) return;

	float tmp0 = x[p[index]] - cx;
	float tmp1 = y[p[index]] - cy;
	float tmp2 = z[p[index]] - cz;

	float coef[9] = { tmp0 * rx[index], tmp0 * ry[index], tmp0 * rz[index],
					  tmp1 * rx[index], tmp1 * ry[index], tmp1 * rz[index],
					  tmp2 * rx[index], tmp2 * ry[index], tmp2 * rz[index] };

	for (int i = 0; i < 9; i++)
	{
		atomicAdd(&A[i], coef[i]);
	}
}

RigidBodyConstraint::~RigidBodyConstraint()
{
	gpuErrchk(cudaFree(p));
	gpuErrchk(cudaFree(rx));
	gpuErrchk(cudaFree(ry));
	gpuErrchk(cudaFree(rz));
}

RigidBodyConstraint RigidBodyConstraint::init(float* x, float* y, float* z, float* m, int* p, int n, ConstraintLimitType type, float compliance)
{
	((Constraint*)this)->init(n, compliance, type);
	gpuErrchk(cudaMalloc((void**)&decompostion, 9 * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&(this->p), n * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(this->rx), n * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&(this->ry), n * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&(this->rz), n * sizeof(float)));

	gpuErrchk(cudaMemcpy(this->p, p, n * sizeof(int), cudaMemcpyHostToDevice));

	thrust::device_ptr<float> x_ptr = thrust::device_pointer_cast(x);
	thrust::device_ptr<float> y_ptr = thrust::device_pointer_cast(y);
	thrust::device_ptr<float> z_ptr = thrust::device_pointer_cast(z);


	thrust::device_ptr<float> m_ptr = thrust::device_pointer_cast(m);

	thrust::device_ptr<float> rx_ptr = thrust::device_pointer_cast(rx);
	thrust::device_ptr<float> ry_ptr = thrust::device_pointer_cast(ry);
	thrust::device_ptr<float> rz_ptr = thrust::device_pointer_cast(rz);


	float totalMass = 0.0f;
	for (int i = 0; i < n; i++)
	{
		cx += x_ptr[p[i]] * m_ptr[p[i]];
		cy += y_ptr[p[i]] * m_ptr[p[i]];
		cz += z_ptr[p[i]] * m_ptr[p[i]];
		totalMass += m_ptr[p[i]];
	}
	
	cx /= totalMass;
	cy /= totalMass;
	cz /= totalMass;

	for (int i = 0; i < n; i++)
	{
		rx_ptr[i] = cx - x_ptr[p[i]];
		ry_ptr[i] = cy - y_ptr[p[i]];
		rz_ptr[i] = cz - z_ptr[p[i]];
	}


	return *this;
}


void RigidBodyConstraint::calculateShapeCovariance(float* x, float* y, float* z)
{
	int threads_per_block = 32;
	int blocks = (n + threads_per_block - 1) / threads_per_block;
	float* tmp;

	gpuErrchk(cudaMalloc((void**)&tmp, 9 * sizeof(float)));

	thrust::device_ptr<float> tmp_ptr = thrust::device_pointer_cast(tmp);
	thrust::fill(tmp_ptr, tmp_ptr + 9, 0.0f);

	calculateShapeCovarianceKern << <blocks, threads_per_block >> > (tmp, x, y, z, p, n, rx, ry, rz, cx, cy, cz);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	float A[9] = { 0.0f };
	float Q[9] = { 0.0f };
	float H[9] = { 0.0f };

	gpuErrchk(cudaMemcpy(A, tmp, 9 * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(tmp);
	
	try {

		polar::polar_decomposition(Q, H, A);
	}
	catch (const std::exception& e)
	{
		printf("Error: %s\n", e.what());
	}

	gpuErrchk(cudaMemcpy(decompostion, Q, 9 * sizeof(float), cudaMemcpyHostToDevice));
}

__global__ void calculatePositionChangeKern(float* x, float* y, float* z, int* p, int n, float* rx, float* ry, float* rz, float cx, float cy, float cz, float* decompostion, float* dx, float* dy, float* dz, float invdt)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) return;

	dx[p[index]] += ((decompostion[0] * rx[index] + decompostion[1] * ry[index] + decompostion[2] * rz[index]) + cx - x[p[index]]) * invdt;
	dy[p[index]] += ((decompostion[3] * rx[index] + decompostion[4] * ry[index] + decompostion[5] * rz[index]) + cy - y[p[index]]) * invdt;
	dz[p[index]] += ((decompostion[6] * rx[index] + decompostion[7] * ry[index] + decompostion[8] * rz[index]) + cz - z[p[index]]) * invdt;
}	

void RigidBodyConstraint::calculatePositionChange(float* x, float* y, float* z, float* dx, float* dy, float* dz, float dt)
{
	int threads_per_block = 32;
	int blocks = (n + threads_per_block - 1) / threads_per_block;
	calculatePositionChangeKern << <blocks, threads_per_block >> > (x, y, z, p, n, rx, ry, rz, cx, cy, cz, decompostion, dx, dy, dz, 1/dt);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}
