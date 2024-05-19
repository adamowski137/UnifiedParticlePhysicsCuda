#include "DirectConstraintSolver.cuh"
#include <cuda_runtime.h>
#include "../../../GpuErrorHandling.hpp"
#include <device_launch_parameters.h>
#include "../../../Constants.hpp"
#include "../../../Constraint/ConstraintStorage.cuh"
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

template <typename T>
__global__ void solveConstraintsDirectlyKern(int nConstraints,
	ConstraintArgs args,
	T* constraints)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nConstraints) return;
	constraints[index].directSolve(args);
}

__global__ void applyOffsetKern(int nParticles,
	float* x, float* y, float* z,
	float* dx, float* dy, float* dz,
	int* nConstraintsPerParticle)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;
	const float omega = 1.5f;
	//const float omega = 1.f;
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
	nConstraintsMaxAllocated = 1000;
	gpuErrchk(cudaMalloc((void**)&dev_delta_lambda, nConstraintsMaxAllocated * sizeof(float)));
}

DirectConstraintSolver::~DirectConstraintSolver()
{
	gpuErrchk(cudaFree(dev_nConstraintsPerParticle));
	gpuErrchk(cudaFree(dev_delta_lambda));
}

void DirectConstraintSolver::calculateForces(
	float* x, float* y, float* z, int* mode,
	float* new_x, float* new_y, float* new_z,
	float* invmass, float dt, int iterations)
{
	auto thrust_x = thrust::device_pointer_cast(new_x);
	auto thrust_y = thrust::device_pointer_cast(new_y);
	auto thrust_z = thrust::device_pointer_cast(new_z);

	auto thrust_dx = thrust::device_pointer_cast(dev_dx);
	auto thrust_dy = thrust::device_pointer_cast(dev_dy);
	auto thrust_dz = thrust::device_pointer_cast(dev_dz);

	ConstraintArgsBuilder builder{};

	for (int i = 0; i < iterations; i++)
	{
		builder.initBase(new_x, new_y, new_z, dev_dx, dev_dy, dev_dz, invmass, dev_nConstraintsPerParticle, dt / iterations);
		builder.addOldPosition(x, y, z);
		this->projectConstraints<DistanceConstraint>(builder.build());
		applyOffset(new_x, new_y, new_z);
		this->projectConstraints<SurfaceConstraint>(builder.build());
		applyOffset(new_x, new_y, new_z);
		auto rigidBodyConstraints = ConstraintStorage<RigidBodyConstraint>::Instance.getCpuConstraints();

		gpuErrchk(cudaMemset(dev_dx, 0, sizeof(float) * nParticles));
		gpuErrchk(cudaMemset(dev_dy, 0, sizeof(float) * nParticles));
		gpuErrchk(cudaMemset(dev_dz, 0, sizeof(float) * nParticles));

		for (int i = 0; i < rigidBodyConstraints.size(); i++)
		{
			rigidBodyConstraints[i]->calculateShapeCovariance(new_x, new_y, new_z, invmass);
			rigidBodyConstraints[i]->calculatePositionChange(builder.build());
		}

		thrust::transform(thrust_x, thrust_x + nParticles, thrust_dx, thrust_x, thrust::plus<float>());
		thrust::transform(thrust_y, thrust_y + nParticles, thrust_dy, thrust_y, thrust::plus<float>());
		thrust::transform(thrust_z, thrust_z + nParticles, thrust_dz, thrust_z, thrust::plus<float>());
		builder.clear();
	}
	clearAllConstraints();
}

void DirectConstraintSolver::calculateStabilisationForces(float* x, float* y, float* z, int* phase, float* new_x, float* new_y, float* new_z, float* invmass, float dt, int iterations)
{
	ConstraintArgsBuilder builder{};
	builder.initBase(new_x, new_y, new_z, dev_dx, dev_dy, dev_dz, invmass, dev_nConstraintsPerParticle, dt / iterations);
	for (int i = 0; i < iterations; i++)
	{
		this->projectConstraints<DistanceConstraint>(builder.build());
		applyOffset(x, y, z);
		applyOffset(new_x, new_y, new_z);
		this->projectConstraints<SurfaceConstraint>(builder.build());
		applyOffset(x, y, z);
		applyOffset(new_x, new_y, new_z);
	}
	clearAllConstraints();
}

void DirectConstraintSolver::applyOffset(float* x, float* y, float* z)
{
	int threads = 32;
	int particleBlocks = (nParticles + threads - 1) / threads;
	applyOffsetKern << <particleBlocks, threads >> > (nParticles, x, y, z, dev_dx, dev_dy, dev_dz, dev_nConstraintsPerParticle);
}

template<typename T>
void DirectConstraintSolver::projectConstraints(ConstraintArgs args)
{
	cudaMemset(dev_dx, 0, sizeof(float) * nParticles);
	cudaMemset(dev_dy, 0, sizeof(float) * nParticles);
	cudaMemset(dev_dz, 0, sizeof(float) * nParticles);
	cudaMemset(dev_nConstraintsPerParticle, 0, sizeof(int) * nParticles);

	auto constraintData = ConstraintStorage<T>::Instance.getConstraints();

	if (constraintData.second > 0)
	{
		int threads = 32;
		int blocks = (constraintData.second + threads - 1) / threads;
		int particleBlocks = (nParticles + threads - 1) / threads;
		solveConstraintsDirectlyKern << <blocks, threads >> > (constraintData.second, args, constraintData.first);
	}
}


