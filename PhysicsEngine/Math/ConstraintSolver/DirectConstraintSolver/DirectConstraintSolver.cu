#include "DirectConstraintSolver.cuh"
#include <cuda_runtime.h>
#include "../../../GpuErrorHandling.hpp"
#include <device_launch_parameters.h>
#include "../../../Constants.hpp"
#include "../../../Constraint/ConstraintStorage.cuh"

template <typename T>
__global__ void solveConstraintsDirectlyKern(int nConstraints,
	float* x, float* y, float* z,
	float* dx, float* dy, float* dz,
	float* invmass, int* nConstraintsPerParticle,
	T* constraints)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nConstraints) return;
	constraints[index].directSolve(x, y, z, dx, dy, dz, invmass, nConstraintsPerParticle);
}

__global__ void applyOffset(int nParticles,
	float* x, float* y, float* z,
	float* dx, float* dy, float* dz,
	int* nConstraintsPerParticle)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;
	const float omega = 1.5f;
	if (nConstraintsPerParticle[index] > 0)
	{
		x[index] += omega * dx[index] / nConstraintsPerParticle[index];
		y[index] += omega * dy[index] / nConstraintsPerParticle[index];
		z[index] += omega * dz[index] / nConstraintsPerParticle[index];
	}
}


DirectConstraintSolver::DirectConstraintSolver(int nParticles) : ConstraintSolver(nParticles)
{
	gpuErrchk(cudaMalloc((void**)&dev_nConstraintsPerParticle, nParticles * sizeof(float)));
}

DirectConstraintSolver::~DirectConstraintSolver()
{
	gpuErrchk(cudaFree(dev_nConstraintsPerParticle));
}

void DirectConstraintSolver::calculateForces(float* new_x, float* new_y, float* new_z, float* invmass, int* phase, float dt, int iterations)
{
	this->projectConstraints<DistanceConstraint>(new_x, new_y, new_z, invmass, phase, dt, iterations);
	this->projectConstraints<SurfaceConstraint>(new_x, new_y, new_z, invmass, phase, dt, iterations);
	clearAllConstraints();
}

void DirectConstraintSolver::calculateStabilisationForces(float* x, float* y, float* z, int* mode, float* new_x, float* new_y, float* new_z, float* invmass, float dt, int iterations)
{
	throw -1;
}

template<typename T>
void DirectConstraintSolver::projectConstraints(float* x, float* y, float* z, float* invmass, int* phase, float dt, int iterations)
{
	cudaMemset(dev_dx, 0, sizeof(float) * nParticles);
	cudaMemset(dev_dy, 0, sizeof(float) * nParticles);
	cudaMemset(dev_dz, 0, sizeof(float) * nParticles);
	cudaMemset(dev_nConstraintsPerParticle, 0, sizeof(int) * nParticles);

	auto constraintData = ConstraintStorage<T>::Instance.getConstraints(true);

	int threads = 32;
	int blocks = (constraintData.second + threads - 1) / threads;
	int particleBlocks = (nParticles + threads - 1) / threads;
	if (constraintData.second > 0)
	{
		solveConstraintsDirectlyKern << <blocks, threads >> > (constraintData.second, x, y, z, dev_dx, dev_dy, dev_dz, invmass, dev_nConstraintsPerParticle, constraintData.first);
		applyOffset << <particleBlocks, threads >> > (nParticles, x, y, z, dev_dx, dev_dy, dev_dz, dev_nConstraintsPerParticle);
	}
}


