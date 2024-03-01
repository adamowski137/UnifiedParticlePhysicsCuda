#include "ConstrainSolver.cuh"
#include "../GpuErrorHandling.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fillJacobiansKern(
	int constrainsAmount, int particles,
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz,
	float* jacobian, float* velocity_jacobian,
	Constrain** constrains)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= constrainsAmount) return;
	Constrain* c = constrains[index];
	for (int i = 0; i < c->n; i++)
	{
		jacobian[i * 3 * particles + c->dev_indexes[i]] = c->positionDerivative(x, y, z, vx, vy, vz, i);
		velocity_jacobian[i * 3 * particles + c->dev_indexes[i]] = c->timePositionDerivative(x, y, z, vx, vy, vz, i);
	}
}

ConstrainSolver::ConstrainSolver(int particles, int constrainsNumber, Constrain** constrains) : particles{particles}
{
	gpuErrchk(cudaMalloc((void**)&dev_constrains, constrainsNumber * sizeof(Constrain*)));
	gpuErrchk(cudaMemcpy(dev_constrains, constrains, constrainsNumber * sizeof(Constrain*), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc((void**)&dev_jacobian, 3 * particles * constrainsNumber * sizeof(float)));
	gpuErrchk(cudaMemset(dev_jacobian, 0,  3 * particles * constrainsNumber * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_velocity_jacobian, 3 * particles * constrainsNumber * sizeof(float)));
	gpuErrchk(cudaMemset(dev_velocity_jacobian, 0, 3 * particles * constrainsNumber * sizeof(float)));
}

ConstrainSolver::~ConstrainSolver()
{
	gpuErrchk(cudaFree(dev_constrains));
	gpuErrchk(cudaFree(dev_jacobian));
	gpuErrchk(cudaFree(dev_velocity_jacobian));
}

void ConstrainSolver::fillJacobians(
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz
	)
{
	int threads = 512;
	int blocks = ceilf(constrains.size() / (float)threads);

	fillJacobiansKern << <threads, blocks >> > (constrainsNumber, particles,
		x, y, z,
		vx, vy, vz, 
		dev_jacobian, dev_velocity_jacobian,
		dev_constrains);
}
