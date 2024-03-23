#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <thrust/device_ptr.h>
#include "../Constrain/ConstrainStorage.cuh"
#include "../Constrain/DistanceConstrain/DistanceConstrain.cuh"
#include "../Constrain/SurfaceConstraint/SurfaceConstraint.cuh"
#include "../List/List.cuh"
#include "LinearSolver.cuh"


__global__ void transposeKern(int columns, int rows, float* A, float* AT);
__global__ void massVectorMultpilyKern(int columns, int rows, float* invMass, float* J);
__global__ void applyForce(float* new_lambda, float* jacobi_transposed, float* fc, int nParticles, int nConstraints);
__global__ void matrixMulKern(const float* a, const float* b, float* c, int N, int K);

template<typename T>
__global__ void fillJacobiansKern(
	int nConstraints, int nParticles,
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz,
	float* jacobian, float* velocity_jacobian,
	T* constrains, ConstrainType type)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nConstraints) return;
	if (type == ConstrainType::DISTANCE)
	{
		(constrains[index]).positionDerivative(x, y, z, vx, vy, vz, 0, &jacobian[index * 3 * nParticles + 3 * (constrains[index]).p[0]]);
		(constrains[index]).timePositionDerivative(x, y, z, vx, vy, vz, 0, &velocity_jacobian[index * 3 * nParticles + 3 * (constrains[index]).p[0]]);

		(constrains[index]).positionDerivative(x, y, z, vx, vy, vz, 1, &jacobian[index * 3 * nParticles + 3 * (constrains[index]).p[1]]);
		(constrains[index]).timePositionDerivative(x, y, z, vx, vy, vz, 1, &velocity_jacobian[index * 3 * nParticles + 3 * (constrains[index]).p[1]]);
	}
	if (type == ConstrainType::SURFACE)
	{
		(constrains[index]).positionDerivative(x, y, z, vx, vy, vz, 0, &jacobian[index * 3 * nParticles + 3 * (constrains[index]).p[0]]);
		(constrains[index]).timePositionDerivative(x, y, z, vx, vy, vz, 0, &velocity_jacobian[index * 3 * nParticles + 3 * (constrains[index]).p[0]]);
	}
}


template <typename T>
__global__ void fillResultVectorKern(int particles, int constrainsNumber, float* b,
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz,
	float* jacobian, float dt,
	float* dev_c_min, float* dev_c_max,
	T* constrains)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= constrainsNumber) return;
	b[index] = -(constrains[index])(x, y, z, vx, vy, vz);
	dev_c_max[index] = constrains[index].cMax;
	dev_c_min[index] = constrains[index].cMin;
}

class ConstrainSolver {
public:
	ConstrainSolver(int particles);
	~ConstrainSolver();
	void calculateForces(
		float* x, float* y, float* z,
		float* new_x, float* new_y, float* new_z,
		float* vx, float* vy, float* vz,
		float* invmass, float* fc, float dt
	);

	void setStaticConstraints(std::vector<std::pair<int, int>> pairs, float d);
	void addDynamicConstraints(List* collisions, int* counts, float d, ConstraintLimitType type);
	void addSurfaceConstraints(SurfaceConstraint* surfaceConstraints, int nSurfaceConstraints);

private:
	// J matrix, dynamically created in every iteration
	float* dev_jacobian;
	float* dev_jacobian_transposed;

	// J dot matrix
	float* dev_velocity_jacobian;

	// as in Ax = b matrix equation
	float* dev_A;
	float* dev_b;

	float* dev_c_min;
	float* dev_c_max;

	// coefficients, results of matrix equation, multiplied by J give force that has to be applied to particles
	float* dev_lambda;
	float* dev_new_lambda;

	int nParticles;

	// if number of constraints decreased between simulation steps we do not want to reallocate arrays
	int nConstraintsMaxAllocated;

	// mainly collision constraints

	SurfaceConstraint* dev_surfaceConstraints;

	void allocateArrays(int size);
	
	template<typename T>
	void projectConstraints(float* fc, float* invmass, float* x, float* y, float* z, float* vx, float* vy, float* vz, float dt, ConstrainType type);

};




template<typename T>
void ConstrainSolver::projectConstraints(float* fc, float* invmass, float* x, float* y, float* z, float* vx, float* vy, float* vz, float dt, ConstrainType type)
{
	std::pair<T*, int> constraints = ConstrainStorage::Instance.getConstraints<T>(type);
	int nConstraints = constraints.second;
	if (nConstraints == 0) return;
	this->allocateArrays(nConstraints);

	unsigned int threads = 32;

	// kernels bound by number of constraints
	int constraint_bound_blocks = (nConstraints + threads - 1) / threads;

	// kernels bound by the size of Jacobian
	int jacobian_bound_blocks = ((3 * nParticles * nConstraints) + threads - 1) / threads;

	int particlex3_bound_blocks = ((3 * nParticles) + threads - 1) / threads;

	int particle_bound_blocks = (nParticles + threads - 1) / threads;
	//fillJacobiansKern<T><<<constraint_bound_blocks, threads>>>(nConstraints, nParticles,
	//	x, y, z,
	//	vx, vy, vz,
	//	dev_jacobian, dev_velocity_jacobian,
	//	constraints.first, type);

	//gpuErrchk(cudaGetLastError());
	//gpuErrchk(cudaDeviceSynchronize());

	//fillResultVectorKern<T> <<<constraint_bound_blocks, threads>>> (nParticles, nConstraints, dev_b,
	//	x, y, z,
	//	vx, vy, vz, dev_jacobian, dt,
	//	dev_c_min, dev_c_max,
	//	constraints.first);

	//gpuErrchk(cudaGetLastError());
	//gpuErrchk(cudaDeviceSynchronize());

	transposeKern <<<jacobian_bound_blocks, threads>>> (
		3 * nParticles,
		nConstraints,
		dev_jacobian,
		dev_jacobian_transposed);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize())

	//massVectorMultpilyKern <<<jacobian_bound_blocks, threads>>> (
	//	3 * nParticles,
	//	nConstraints,
	//	invmass,
	//	dev_jacobian);

	//gpuErrchk(cudaGetLastError());
	//gpuErrchk(cudaDeviceSynchronize());

	//unsigned int BLOCKS_X = (nConstraints + threads - 1) / threads;
	//unsigned int BLOCKS_Y = (nConstraints + threads - 1) / threads;

	//dim3 t{ threads, threads };
	//dim3 b{ BLOCKS_X, BLOCKS_Y };

	//matrixMulKern<<<b, t>>> (dev_jacobian, dev_jacobian_transposed, dev_A, nConstraints, 3 * nParticles);

	//gpuErrchk(cudaGetLastError());
	//gpuErrchk(cudaDeviceSynchronize());

	//jaccobi(nConstraints, dev_A, dev_b, dev_lambda, dev_new_lambda, dev_c_min, dev_c_max, 1);

	//applyForce <<<particlex3_bound_blocks, threads>>> (dev_new_lambda, dev_jacobian_transposed, fc, nParticles, nConstraints);

	//gpuErrchk(cudaGetLastError());
	//gpuErrchk(cudaDeviceSynchronize());
}
