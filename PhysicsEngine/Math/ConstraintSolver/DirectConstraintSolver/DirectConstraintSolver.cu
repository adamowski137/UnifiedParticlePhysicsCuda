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

__global__ void setConstraintsKern(int cardinality, int* nConstraintsPerParticle, int* p)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= cardinality) return;
	nConstraintsPerParticle[p[index]] = 1;
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
	thrust::device_ptr<float> dev_dx_ptr(dev_dx);
	thrust::device_ptr<float> dev_dy_ptr(dev_dy);
	thrust::device_ptr<float> dev_dz_ptr(dev_dz);

	ConstraintArgsBuilder builder{};
	builder.initBase(new_x, new_y, new_z, dev_dx, dev_dy, dev_dz, invmass, dev_nConstraintsPerParticle, dt / iterations);
	builder.addOldPosition(x, y, z);

	for (int i = 0; i < iterations; i++)
	{
		this->projectConstraints<DistanceConstraint>(builder.build());
		applyOffset(new_x, new_y, new_z);
		this->projectConstraints<SurfaceConstraint>(builder.build());
		applyOffset(new_x, new_y, new_z);
		auto rigidBodyConstraints = ConstraintStorage<RigidBodyConstraint>::Instance.getCpuConstraints();

		gpuErrchk(cudaMemset(dev_dx, 0, sizeof(float) * nParticles));
		gpuErrchk(cudaMemset(dev_dy, 0, sizeof(float) * nParticles));
		gpuErrchk(cudaMemset(dev_dz, 0, sizeof(float) * nParticles));
		gpuErrchk(cudaMemset(dev_nConstraintsPerParticle, 0, sizeof(int) * nParticles));

		for (int i = 0; i < rigidBodyConstraints.size(); i++)
		{
			int threads = 32;
			int blocks = (rigidBodyConstraints[i]->n + threads - 1) / threads;
			setConstraintsKern<<<blocks, threads>>>(rigidBodyConstraints[i]->n, dev_nConstraintsPerParticle, rigidBodyConstraints[i]->p);
			gpuErrchk(cudaGetLastError());
			gpuErrchk(cudaDeviceSynchronize());

			rigidBodyConstraints[i]->calculateShapeCovariance(new_x, new_y, new_z, invmass);
			rigidBodyConstraints[i]->calculatePositionChange(builder.build());
		}

		applyOffset(new_x, new_y, new_z);
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
	gpuErrchk(cudaMemset(dev_dx, 0, sizeof(float) * nParticles));
	gpuErrchk(cudaMemset(dev_dy, 0, sizeof(float) * nParticles));
	gpuErrchk(cudaMemset(dev_dz, 0, sizeof(float) * nParticles));
	gpuErrchk(cudaMemset(dev_nConstraintsPerParticle, 0, sizeof(int) * nParticles));

	auto constraintData = ConstraintStorage<T>::Instance.getConstraints();

	if (constraintData.second > 0)
	{
		int threads = 32;
		int blocks = (constraintData.second + threads - 1) / threads;
		int particleBlocks = (nParticles + threads - 1) / threads;
		solveConstraintsDirectlyKern << <blocks, threads >> > (constraintData.second, args, constraintData.first);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}
}


