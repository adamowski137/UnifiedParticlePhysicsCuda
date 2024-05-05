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

	float rnx = x[p[index]] - cx;
	float rny = y[p[index]] - cy;
	float rnz = z[p[index]] - cz;

	float coef[9] = { rnx * rx[index], rnx * ry[index], rnx * rz[index],
					  rny * rx[index], rny * ry[index], rny * rz[index],
					  rnz * rx[index], rnz * ry[index], rnz * rz[index] };

	for (int i = 0; i < 9; i++)
	{
		atomicAdd(&A[i], coef[i]);
	}
}

RigidBodyConstraint::~RigidBodyConstraint()
{
	if (p != 0)
		gpuErrchk(cudaFree(p));
	if (rx != 0)
		gpuErrchk(cudaFree(rx));
	if (ry != 0)
		gpuErrchk(cudaFree(ry));
	if (rz != 0)
		gpuErrchk(cudaFree(rz));
	if (decompostion != 0)
		gpuErrchk(cudaFree(decompostion));
}

RigidBodyConstraint::RigidBodyConstraint(float* x, float* y, float* z, float* invm, int* p, int n, ConstraintLimitType type, float compliance)
{
	((Constraint*)this)->init(n, compliance, type);
	gpuErrchk(cudaMalloc((void**)&decompostion, 9 * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&(this->p), n * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(this->rx), n * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&(this->ry), n * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&(this->rz), n * sizeof(float)));

	gpuErrchk(cudaMemcpy(this->p, p, n * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(this->p, p, n * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(this->p, p, n * sizeof(int), cudaMemcpyHostToDevice));


	thrust::device_ptr<float> x_ptr = thrust::device_pointer_cast(x);
	thrust::device_ptr<float> y_ptr = thrust::device_pointer_cast(y);
	thrust::device_ptr<float> z_ptr = thrust::device_pointer_cast(z);


	thrust::device_ptr<float> invm_ptr = thrust::device_pointer_cast(invm);

	thrust::device_ptr<float> rx_ptr = thrust::device_pointer_cast(rx);
	thrust::device_ptr<float> ry_ptr = thrust::device_pointer_cast(ry);
	thrust::device_ptr<float> rz_ptr = thrust::device_pointer_cast(rz);


	float totalMass = 0.0f;
	cx = 0;
	cy = 0;
	cz = 0;
	for (int i = 0; i < n; i++)
	{
		cx += x_ptr[p[i]] / invm_ptr[p[i]];
		cy += y_ptr[p[i]] / invm_ptr[p[i]];
		cz += z_ptr[p[i]] / invm_ptr[p[i]];
		totalMass += 1 / invm_ptr[p[i]];
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


bool RigidBodyConstraint::calculateShapeCovariance(float* x, float* y, float* z)
{
	int threads_per_block = 32;
	int blocks = (n + threads_per_block - 1) / threads_per_block;
	float* tmp;

	gpuErrchk(cudaMalloc((void**)&tmp, 9 * sizeof(float)));
	gpuErrchk(cudaMemset(tmp, 0, 9 * sizeof(float)));

	calculateShapeCovarianceKern << <blocks, threads_per_block >> > (tmp, x, y, z, p, n, rx, ry, rz, cx, cy, cz);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	float A[9] = { 0.0f };
	float Q[9] = { 0.0f };
	float H[9] = { 0.0f };

	gpuErrchk(cudaMemcpy(A, tmp, 9 * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(tmp));
	
	if (std::isnan(A[0]))
	{
		std::cout << "A is nan" << std::endl;
		return false;
	}	

	if (std::isinf(A[0]))
	{
		std::cout << "A is inf" << std::endl;
		return false;
	}

	polar::polar_decomposition(Q, H, A);

	gpuErrchk(cudaMemcpy(decompostion, Q, 9 * sizeof(float), cudaMemcpyHostToDevice));

	return true;
}

__global__ void calculatePositionChangeKern(float* x, float* y, float* z, int* p, int n, float* rx, float* ry, float* rz, float cx, float cy, float cz, float* decompostion, float* dx, float* dy, float* dz, float dt)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) return;
	// decompostion is column major
	dx[p[index]] += ((decompostion[0] * rx[index] + decompostion[3] * ry[index] + decompostion[6] * rz[index]) + cx - x[p[index]]);
	dy[p[index]] += ((decompostion[1] * rx[index] + decompostion[4] * ry[index] + decompostion[7] * rz[index]) + cy - y[p[index]]);
	dz[p[index]] += ((decompostion[2] * rx[index] + decompostion[5] * ry[index] + decompostion[8] * rz[index]) + cz - z[p[index]]);
}	

void RigidBodyConstraint::calculatePositionChange(float* x, float* y, float* z, float* dx, float* dy, float* dz, float dt)
{
	int threads_per_block = 32;
	int blocks = (n + threads_per_block - 1) / threads_per_block;
	calculatePositionChangeKern << <blocks, threads_per_block >> > (x, y, z, p, n, rx, ry, rz, cx, cy, cz, decompostion, dx, dy, dz, dt);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}
