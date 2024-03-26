#include "ConstrainSolver.cuh"
#include "../GpuErrorHandling.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "../Constants.hpp"

#define SHMEM_SIZE 1024


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


__global__ void matrixMulKern(const float* a, const float* b, float* c, int N, int K) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float s_a[SHMEM_SIZE];
	__shared__ float s_b[SHMEM_SIZE];

	float tmp = 0;

	for (int i = 0; i < K; i += blockDim.x)
	{

		s_a[threadIdx.y * blockDim.x + threadIdx.x] = 0;
		s_b[threadIdx.y * blockDim.x + threadIdx.x] = 0;
		__syncthreads();


		if (row < N && i + threadIdx.x < K)
			s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i + threadIdx.x];
		if (col < N && i + threadIdx.y < K)
			s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];
		__syncthreads();

		if (row < N && col < N)
		{
			for (int j = 0; j < blockDim.x; j++) {
				tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
			}
		}
		__syncthreads();
	}

	if (row < N && col < N)
		c[row * N + col] = tmp;
}

__global__ void massVectorMultpilyKern(int columns, int rows, float* invMass, float* J)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= columns * rows) return;
	int column = index % columns;
	J[index] *= invMass[column / 3];
}

__global__ void transposeKern(int columns, int rows, float* A, float* AT)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= columns * rows) return;
	int column = index % columns;
	int row = index / columns;

	AT[column * rows + row] = A[row * columns + column];
}


__global__ void applyForce(float* new_lambda, float* jacobi_transposed, float* fc, int nParticles, int nConstraints)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < 3 * nParticles)
	{
		for (int i = 0; i < nConstraints; i++)
		{
			fc[index] += new_lambda[i] * jacobi_transposed[index * nConstraints + i];
		}
	}
}

template<typename T>
void fillJacobiansWrapper(int nConstraints, int nParticles,
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz,
	float* jacobian, float* velocity_jacobian,
	float* jacobian_transposed, float* A,
	float* b, float dt,
	float* invmass, float* fc, float* lambda, float* new_lambda, float* c_min, float* c_max,
	T* constraints, ConstrainType type)
{
	unsigned int threads = 32;

	// kernels bound by number of constraints
	int constraint_bound_blocks = (nConstraints + threads - 1) / threads;

	// kernels bound by the size of Jacobian
	int jacobian_bound_blocks = ((3 * nParticles * nConstraints) + threads - 1) / threads;

	int particlex3_bound_blocks = ((3 * nParticles) + threads - 1) / threads;

	int particle_bound_blocks = (nParticles + threads - 1) / threads;

	fillJacobiansKern<<<constraint_bound_blocks, threads>>>(nConstraints, nParticles,
		x, y, z,
		vx, vy, vz,
		jacobian, velocity_jacobian,
		constraints, type);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillResultVectorKern << <constraint_bound_blocks, threads >> > (nParticles, nConstraints, b,
		x, y, z,
		vx, vy, vz, jacobian, dt,
		c_min, c_max,
		constraints);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	massVectorMultpilyKern <<<jacobian_bound_blocks, threads>>> (
		3 * nParticles,
		nConstraints,
		invmass,
		jacobian);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	unsigned int BLOCKS_X = (nConstraints + threads - 1) / threads;
	unsigned int BLOCKS_Y = (nConstraints + threads - 1) / threads;

	dim3 th{ threads, threads };
	dim3 bl{ BLOCKS_X, BLOCKS_Y };

	matrixMulKern<<<bl, th>>> (jacobian, jacobian_transposed, A, nConstraints, 3 * nParticles);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	jaccobi(nConstraints, A, b, lambda, new_lambda, c_min, c_max, 1);

	applyForce <<<particlex3_bound_blocks, threads>>> (new_lambda, jacobian_transposed, fc, nParticles, nConstraints);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

ConstrainSolver::ConstrainSolver(int particles) : nParticles{ particles }
{
	// set pointers to 0 - this way it will be easy to distinguish whether they have already been allocated or not
	dev_jacobian = 0;
	dev_jacobian_transposed = 0;
	dev_velocity_jacobian = 0;
	dev_A = 0;
	dev_b = 0;
	dev_lambda = 0;
	dev_new_lambda = 0;
	dev_c_min = 0;
	dev_c_max = 0;

	ConstrainStorage::Instance.initInstance();
}

ConstrainSolver::~ConstrainSolver()
{
	gpuErrchk(cudaFree(dev_jacobian));
	gpuErrchk(cudaFree(dev_jacobian_transposed));
	gpuErrchk(cudaFree(dev_velocity_jacobian));
	gpuErrchk(cudaFree(dev_A));
	gpuErrchk(cudaFree(dev_b));
	gpuErrchk(cudaFree(dev_lambda));
	gpuErrchk(cudaFree(dev_new_lambda));
	gpuErrchk(cudaFree(dev_c_min));
	gpuErrchk(cudaFree(dev_c_max));
}

void ConstrainSolver::calculateForces(
	float* x, float* y, float* z,
	float* new_x, float* new_y, float* new_z,
	float* vx, float* vy, float* vz,
	float* invmass, float* fc, float dt
)
{
	this->projectConstraints<DistanceConstrain>(fc, invmass, x, y, z, vx, vy, vz, dt, ConstrainType::DISTANCE);
	this->projectConstraints<SurfaceConstraint>(fc, invmass, x, y, z, vx, vy, vz, dt, ConstrainType::SURFACE);
}

void ConstrainSolver::setStaticConstraints(std::vector<std::pair<int, int>> pairs, float d)
{
	std::vector<DistanceConstrain> cpu_constraints;
	for (const auto& pair : pairs)
	{
		cpu_constraints.push_back(DistanceConstrain().init(d, pair.first, pair.second, ConstraintLimitType::EQ));
	}

	ConstrainStorage::Instance.setStaticConstraints<DistanceConstrain>(cpu_constraints.data(), cpu_constraints.size(), ConstrainType::DISTANCE);

}

void ConstrainSolver::addDynamicConstraints(List* collisions, int* sums, float d, ConstraintLimitType type)
{
	ConstrainStorage::Instance.addCollisions(collisions, sums, type, d, nParticles);
}

void ConstrainSolver::addSurfaceConstraints(SurfaceConstraint* surfaceConstraints, int nSurfaceConstraints)
{
	this->dev_surfaceConstraints = surfaceConstraints;
}

void ConstrainSolver::allocateArrays(int nConstraints)
{
	if (nConstraints > nConstraintsMaxAllocated)
	{
		if (dev_jacobian_transposed != 0)
			gpuErrchk(cudaFree(dev_jacobian_transposed));
		gpuErrchk(cudaMalloc((void**)&dev_jacobian_transposed, 3 * nParticles * nConstraints * sizeof(float)));
		gpuErrchk(cudaMemset(dev_jacobian_transposed, 0, 3 * nParticles * nConstraints * sizeof(float)));

		if (dev_velocity_jacobian != 0)
			gpuErrchk(cudaFree(dev_velocity_jacobian));
		gpuErrchk(cudaMalloc((void**)&dev_velocity_jacobian, 3 * nParticles * nConstraints * sizeof(float)));
		gpuErrchk(cudaMemset(dev_velocity_jacobian, 0, 3 * nParticles * nConstraints * sizeof(float)));

		if (dev_A != 0)
			gpuErrchk(cudaFree(dev_A));
		gpuErrchk(cudaMalloc((void**)&dev_A, nConstraints * nConstraints * sizeof(float)));
		gpuErrchk(cudaMemset(dev_A, 0, nConstraints * nConstraints * sizeof(float)));

		if (dev_b != 0)
			gpuErrchk(cudaFree(dev_b));
		gpuErrchk(cudaMalloc((void**)&dev_b, nConstraints * sizeof(float)));
		gpuErrchk(cudaMemset(dev_b, 0, nConstraints * sizeof(float)));

		if (dev_lambda != 0)
			gpuErrchk(cudaFree(dev_lambda));
		gpuErrchk(cudaMalloc((void**)&dev_lambda, nConstraints * sizeof(float)));
		gpuErrchk(cudaMemset(dev_lambda, 0, nConstraints * sizeof(float)));

		if (dev_new_lambda != 0)
			gpuErrchk(cudaFree(dev_new_lambda));
		gpuErrchk(cudaMalloc((void**)&dev_new_lambda, nConstraints * sizeof(float)));
		gpuErrchk(cudaMemset(dev_new_lambda, 0, nConstraints * sizeof(float)));


		if (dev_c_min != 0)
			gpuErrchk(cudaFree(dev_c_min));
		gpuErrchk(cudaMalloc((void**)&dev_c_min, nConstraints * sizeof(float)));
		gpuErrchk(cudaMemset(dev_c_min, 0, nConstraints * sizeof(float)));

		if (dev_c_max != 0)
			gpuErrchk(cudaFree(dev_c_max));
		gpuErrchk(cudaMalloc((void**)&dev_c_max, nConstraints * sizeof(float)));
		gpuErrchk(cudaMemset(dev_c_max, 0, nConstraints * sizeof(float)));

	}
}


