#include "DirectConstraintSolverGaussSeidel.cuh"
#include <cuda_runtime.h>
#include "../../../GpuErrorHandling.hpp"
#include <device_launch_parameters.h>
#include "../../../Constants.hpp"
#include "../../../Constraint/ConstraintStorage.cuh"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <set>

template<typename T>
__global__ void solveConstraintGaussSeidel(int nConstraints, ConstraintArgs args, T* constraints, int* coloring, int nColors)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nConstraints) return;

	for (int color = 0; color < nColors; color++)
	{
		if (coloring[index] == color)
		{
			constraints[index].directSolve_GaussSeidel(args);
		}
	}
}

DirectConstraintSolverGaussSeidel::DirectConstraintSolverGaussSeidel(int nParticles) 
	: ConstraintSolver(nParticles), nConstraintsMaxAllocated(1000)
{
	gpuErrchk(cudaMalloc((void**)&dev_coloring, sizeof(int) * nConstraintsMaxAllocated));
}

DirectConstraintSolverGaussSeidel::~DirectConstraintSolverGaussSeidel()
{
	gpuErrchk(cudaFree(dev_coloring));
}

void DirectConstraintSolverGaussSeidel::calculateForces(float* x, float* y, float* z, int* mode, float* new_x, float* new_y, float* new_z, float* invmass, float dt, int iterations)
{
	auto thrust_x = thrust::device_pointer_cast(new_x);
	auto thrust_y = thrust::device_pointer_cast(new_y);
	auto thrust_z = thrust::device_pointer_cast(new_z);

	auto thrust_dx = thrust::device_pointer_cast(dev_dx);
	auto thrust_dy = thrust::device_pointer_cast(dev_dy);
	auto thrust_dz = thrust::device_pointer_cast(dev_dz);

	ConstraintArgsBuilder builder{};
	builder.initBase(new_x, new_y, new_z, dev_dx, dev_dy, dev_dz, invmass, nullptr, dt / iterations);
	builder.addOldPosition(x, y, z);

	for (int i = 0; i < iterations; i++)
	{
		this->projectConstraints<DistanceConstraint>(builder.build());
		this->projectConstraints<SurfaceConstraint>(builder.build());
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
	}
	clearAllConstraints();
}

void DirectConstraintSolverGaussSeidel::calculateStabilisationForces(float* x, float* y, float* z, int* mode, float* new_x, float* new_y, float* new_z, float* invmass, float dt, int iterations)
{

}

template<>
int DirectConstraintSolverGaussSeidel::getParticlePerConstraint<DistanceConstraint>()
{
	return 2;
}

template<>
int DirectConstraintSolverGaussSeidel::getParticlePerConstraint<SurfaceConstraint>()
{
	return 1;
}

template<typename T>
int DirectConstraintSolverGaussSeidel::findColoring(std::pair<T*, int> constraintData)
{
	if (constraintData.second > nConstraintsMaxAllocated)
		reallocateColoring(constraintData.second);

	int particlePerConstraint = getParticlePerConstraint<T>();
	auto coloringPtr = thrust::device_pointer_cast(dev_coloring);

	std::vector<T> cpu_constraints(constraintData.second);
	gpuErrchk(cudaMemcpy(cpu_constraints.data(), constraintData.first, sizeof(T) * constraintData.second, cudaMemcpyDeviceToHost));



	thrust::fill(coloringPtr, coloringPtr + nConstraintsMaxAllocated, -1);

	std::set<int> particles;
	int color = 0;
	do {
		particles.clear();
		for (int i = 0; i < constraintData.second; i++)
		{
			if (coloringPtr[i] != -1)
				continue;

			std::vector<int> particleIndicies;
			bool nextColor = false;
			
			for (int j = 0; j < particlePerConstraint; j++)
				if (particles.find(cpu_constraints[i].p[j]) != particles.end())
					nextColor = true;
			
			if (nextColor)
				continue;
			
			for (int j = 0; j < particlePerConstraint; j++)
				particles.insert(cpu_constraints[i].p[j]);

			coloringPtr[i] = color;
		}

		color++;
	} while (!particles.empty());

	//for (int i = 0; i < constraintData.second; i++)
	//	std::cout << coloringPtr[i] << " ";


	return color;
}

void DirectConstraintSolverGaussSeidel::reallocateColoring(int n)
{
	while (nConstraintsMaxAllocated < n)
		nConstraintsMaxAllocated <<= 1;
	gpuErrchk(cudaFree(dev_coloring));
	gpuErrchk(cudaMalloc((void**)&dev_coloring, sizeof(int) * nConstraintsMaxAllocated));
}

template<typename T>
void DirectConstraintSolverGaussSeidel::projectConstraints(ConstraintArgs args)
{
	auto constraintData = ConstraintStorage<T>::Instance.getConstraints();

	if (constraintData.second > 0)
	{
		int nColors = findColoring(constraintData);
		int threads = 32;
		int blocks = (constraintData.second + threads - 1) / threads;
		solveConstraintGaussSeidel << <blocks, threads >> > (constraintData.second, args, constraintData.first, dev_coloring, nColors);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

}
