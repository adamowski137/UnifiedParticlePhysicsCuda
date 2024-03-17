#include "ConstrainSolver.cuh"
#include "../GpuErrorHandling.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "LinearSolver.cuh"
#include "../Constants.hpp"

#define SHMEM_SIZE 1024

__global__ void fillJacobiansKern(
	int constrainsAmount, int nSurfaceConstraints, int particles,
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz,
	float* jacobian, float* velocity_jacobian,
	DistanceConstrain* constrains)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= constrainsAmount - nSurfaceConstraints) return;

	(constrains[index]).positionDerivative(x, y, z, vx, vy, vz, 0, &jacobian[index * 3 * particles + 3 * (constrains[index]).p1]);
	(constrains[index]).timePositionDerivative(x, y, z, vx, vy, vz, 0, &velocity_jacobian[index * 3 * particles + 3 * (constrains[index]).p1]);

	(constrains[index]).positionDerivative(x, y, z, vx, vy, vz, 1, &jacobian[index * 3 * particles + 3 * (constrains[index]).p2]);
	(constrains[index]).timePositionDerivative(x, y, z, vx, vy, vz, 1, &velocity_jacobian[index * 3 * particles + 3 * (constrains[index]).p2]);
}

__global__ void fillJacobiansWithSurfaceConstraintKern(
	int offset, int nSurfaceConstraints, int nParticles,
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz,
	float* jacobian, float* velocity_jacobian,
	SurfaceConstraint* constrains)
{	
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nSurfaceConstraints) return;

	(constrains[index]).positionDerivative(x, y, z, vx, vy, vz, &jacobian[(offset + index) * 3 * nParticles + 3 * (constrains[index]).p]);
	(constrains[index]).timePositionDerivative(x, y, z, vx, vy, vz, &velocity_jacobian[(offset + index) * 3 * nParticles + 3 * (constrains[index]).p]);
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

__global__ void fillResultVectorKern(int particles, int constrainsNumber, float* b, 
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz,
	float* jacobian, float dt,
	DistanceConstrain* constrains)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= constrainsNumber) return;
	b[index] = -(constrains[index])(x, y, z, vx, vy, vz) - (constrains[index]).timeDerivative(x, y, z, vx, vy, vz);
}

__global__ void fillResultVectorKern(int particles, int constrainsNumber, float* b, 
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz,
	float* jacobian, float dt,
	SurfaceConstraint* constrains)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= constrainsNumber) return;
	b[index] = -(constrains[index])(x, y, z, vx, vy, vz) - (constrains[index]).timeDerivative(x, y, z, vx, vy, vz);
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

__global__ void addCollisionsKern(List* collisions, int* counts, DistanceConstrain* constraints, ConstraintLimitType type, float d, int nParticles)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles - 1) return;
	Node* p = collisions[index].head;
	int constrainIndex = counts[index] - 1;
	
	while (p != NULL)
	{
		constraints[constrainIndex] = DistanceConstrain(d, index, p->value, type);
		p = p->next;
		constrainIndex--;
	}
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
	dev_constraints = 0;
	dev_staticConstraints = 0;

	nDynamicConstraints = 0;
	nStaticConstraints = 0;
	nConstraints = 0;
	
	gpuErrchk(cudaMalloc((void**)&dev_dynamicConstraints, sizeof(DistanceConstrain) * 100));

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


	gpuErrchk(cudaFree(dev_staticConstraints));
	gpuErrchk(cudaFree(dev_constraints));
}

void ConstrainSolver::calculateForces(
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz,
	float* invmass, float* fc, float dt
)
{
	nConstraints = nStaticConstraints + nDynamicConstraints + nSurfaceConstraints;

	unsigned int threads = 32;

	// kernels bound by number of constraints
	int constraint_bound_blocks = (nConstraints + threads - 1) / threads;

	// kernels bound by the size of Jacobian
	int jacobian_bound_blocks = ((3 * nParticles * nConstraints) + threads - 1) / threads;
	
	int particlex3_bound_blocks = ((3 * nParticles) + threads - 1) / threads;

	int particle_bound_blocks = (nParticles + threads - 1) / threads;
	


	this->allocateArrays();
	this->projectConstraints(x, y, z, vx, vy, vz, dt);
	

	transposeKern << <jacobian_bound_blocks, threads>> > (
		3 * nParticles,
		nConstraints,
		dev_jacobian,
		dev_jacobian_transposed);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize())

	massVectorMultpilyKern << <jacobian_bound_blocks, threads >> > (
		3 * nParticles,
		nConstraints,
		invmass,
		dev_jacobian);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	unsigned int BLOCKS_X = (nConstraints + threads - 1) / threads;
	unsigned int BLOCKS_Y = (nConstraints + threads - 1) / threads;

	dim3 t{ threads, threads };
	dim3 b{ BLOCKS_X, BLOCKS_Y };

	matrixMulKern<<<b, t>>>(dev_jacobian, dev_jacobian_transposed, dev_A, nConstraints, 3 * nParticles);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	jaccobiKern << <constraint_bound_blocks, threads >> > (nConstraints, dev_A, dev_b, dev_lambda, dev_new_lambda);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//std::swap(dev_lambda, dev_new_lambda);
	applyForce << <particlex3_bound_blocks, threads >> > (dev_new_lambda, dev_jacobian_transposed, fc, 3 * nParticles, nConstraints);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//thrust::device_ptr<float> f{ fc };
	//for (int i = 0; i < nParticles; i++)
	//{
	//	std::cout << f[3 * i] << " " << f[3 * i + 1] << " " << f[3 * i + 2] << std::endl;
	//}
	nSurfaceConstraints = 0;
	gpuErrchk(cudaFree(dev_surfaceConstraints));
}

void ConstrainSolver::setStaticConstraints(std::vector<std::pair<int, int>> pairs, float d)
{
	nStaticConstraints = pairs.size();
	std::vector<DistanceConstrain> cpu_constraints;
	for (const auto& pair : pairs)
	{
		cpu_constraints.push_back(DistanceConstrain(d, pair.first, pair.second, ConstraintLimitType::EQ));
	}

	gpuErrchk(cudaMalloc(&dev_staticConstraints, sizeof(DistanceConstrain) * nStaticConstraints));
	gpuErrchk(cudaMemcpy(dev_staticConstraints, cpu_constraints.data(), cpu_constraints.size() * sizeof(DistanceConstrain), cudaMemcpyHostToDevice));
}

void ConstrainSolver::addDynamicConstraints(List* collisions, int* sums, float d, ConstraintLimitType type)
{
	int threads = 32;
	int blocks = (nParticles + threads - 1) / threads;
	thrust::device_ptr<int> p = thrust::device_pointer_cast<int>(sums);


	addCollisionsKern<<<blocks, threads >>>(collisions, sums, dev_dynamicConstraints, type, d, nParticles);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	nDynamicConstraints += p[nParticles - 1];
}

void ConstrainSolver::addSurfaceConstraints(SurfaceConstraint* surfaceConstraints, int nSurfaceConstraints)
{
	this->dev_surfaceConstraints = surfaceConstraints;
	this->nSurfaceConstraints = nSurfaceConstraints;
}

void ConstrainSolver::allocateArrays()
{
	if (nConstraints > nConstraintsMaxAllocated)
	{
		if (dev_constraints != 0)
			gpuErrchk(cudaFree(dev_constraints));
		gpuErrchk(cudaMalloc(&dev_constraints, sizeof(DistanceConstrain) * nConstraints));
		gpuErrchk(cudaMemset(dev_constraints, 0, sizeof(DistanceConstrain) * nConstraints));
		gpuErrchk(cudaMemcpy(dev_constraints, dev_staticConstraints, nStaticConstraints * sizeof(DistanceConstrain), cudaMemcpyDeviceToDevice));

		if (dev_jacobian != 0)
			gpuErrchk(cudaFree(dev_jacobian));
		gpuErrchk(cudaMalloc((void**)&dev_jacobian, 3 * nParticles * nConstraints * sizeof(float)));
		gpuErrchk(cudaMemset(dev_jacobian, 0, 3 * nParticles * nConstraints * sizeof(float)));

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

		nConstraintsMaxAllocated = nConstraints;
	}
}


void ConstrainSolver::projectConstraints(float* x, float* y, float* z, float* vx, float* vy, float* vz, float dt)
{
	gpuErrchk(cudaMemcpy(dev_constraints + nStaticConstraints, dev_dynamicConstraints,
		sizeof(DistanceConstrain) * nDynamicConstraints, cudaMemcpyDeviceToDevice));
	int threads = 32;
	int blocks = (nConstraints + threads - 1) / threads;


	fillJacobiansKern << < blocks, threads >> > (nConstraints, nSurfaceConstraints, nParticles,
		x, y, z,
		vx, vy, vz,
		dev_jacobian, dev_velocity_jacobian,
		dev_constraints);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	fillJacobiansWithSurfaceConstraintKern << < blocks, threads >> > (nConstraints - nSurfaceConstraints, nSurfaceConstraints, nParticles,
		x, y, z,
		vx, vy, vz,
		dev_jacobian, dev_velocity_jacobian,
		dev_surfaceConstraints);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());



	fillResultVectorKern<<<blocks, threads>>>(nParticles, nConstraints, dev_b,
		x, y, z,
		vx, vy, vz, dev_jacobian, dt,
		dev_constraints);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	fillResultVectorKern<<<blocks, threads>>>(nParticles, nSurfaceConstraints, dev_b + nConstraints - nSurfaceConstraints,
		x, y, z,
		vx, vy, vz, dev_jacobian, dt,
		dev_surfaceConstraints);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());



	gpuErrchk(cudaFree(dev_surfaceConstraints));
	
	nDynamicConstraints = 0;
}
