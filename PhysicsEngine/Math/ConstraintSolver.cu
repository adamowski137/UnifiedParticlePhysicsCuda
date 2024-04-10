#include "ConstraintSolver.cuh"
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
	float* jacobian,
	T* constrains, ConstraintType type)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nConstraints) return;
	if (type == ConstraintType::DISTANCE)
	{
		(constrains[index]).positionDerivative(x, y, z, vx, vy, vz, 0, &jacobian[index * 3 * nParticles + 3 * (constrains[index]).p[0]]);
		(constrains[index]).positionDerivative(x, y, z, vx, vy, vz, 1, &jacobian[index * 3 * nParticles + 3 * (constrains[index]).p[1]]);
	}
	if (type == ConstraintType::SURFACE)
	{
		(constrains[index]).positionDerivative(x, y, z, vx, vy, vz, 0, &jacobian[index * 3 * nParticles + 3 * (constrains[index]).p[0]]);
	}
}


template <typename T>
__global__ void fillResultVectorKern(int particles, int constrainsNumber, float* b,
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz,
	float* jacobian,
	float* dev_c_min, float* dev_c_max,
	T* constrains)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= constrainsNumber) return;
	b[index] = -5 * (constrains[index])(x, y, z, vx, vy, vz);
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


__global__ void applyForce(float* new_lambda, float* jacobi_transposed, float* dx, float* dy, float* dz, int nParticles, int nConstraints, float dt)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < nParticles)
	{
		float sumX = 0, sumY = 0, sumZ = 0;
		int sumC = 0;
		for (int i = 0; i < nConstraints; i++)
		{
			if (jacobi_transposed[(3 * index + 0) * nConstraints + i] == 0 &&
				jacobi_transposed[(3 * index + 1) * nConstraints + i] == 0 &&
				jacobi_transposed[(3 * index + 2) * nConstraints + i] == 0) continue;
			
			sumC++;
			sumX += new_lambda[i] * jacobi_transposed[(3 * index + 0) * nConstraints + i];
			sumY += new_lambda[i] * jacobi_transposed[(3 * index + 1) * nConstraints + i];
			sumZ += new_lambda[i] * jacobi_transposed[(3 * index + 2) * nConstraints + i];
		}
		if(sumC == 0) return;
		dx[index] += sumX * dt / sumC;
		dy[index] += sumY * dt / sumC;
		dz[index] += sumZ * dt / sumC;
	}
}

template<typename T>
void fillJacobiansWrapper(int nConstraints, int nParticles,
	float* x, float* y, float* z,
	float* dx, float* dy, float* dz,
	float* vx, float* vy, float* vz,
	float* jacobian,
	float* jacobian_transposed, float* A,
	float* b, float dt,
	float* invmass, float* lambda, float* new_lambda, float* c_min, float* c_max,
	T* constraints, ConstraintType type)
{
	unsigned int threads = 32;

	// kernels bound by number of constraints
	int constraint_bound_blocks = (nConstraints + threads - 1) / threads;

	// kernels bound by the size of Jacobian
	int jacobian_bound_blocks = ((3 * nParticles * nConstraints) + threads - 1) / threads;

	int particlex3_bound_blocks = ((3 * nParticles) + threads - 1) / threads;

	int particle_bound_blocks = (nParticles + threads - 1) / threads;

	fillJacobiansKern << <constraint_bound_blocks, threads >> > (nConstraints, nParticles,
		x, y, z,
		vx, vy, vz,
		jacobian,
		constraints, type);

	transposeKern << <jacobian_bound_blocks, threads >> > (
		3 * nParticles,
		nConstraints,
		jacobian,
		jacobian_transposed);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillResultVectorKern << <constraint_bound_blocks, threads >> > (nParticles, nConstraints, b,
		x, y, z,
		vx, vy, vz, jacobian,
		c_min, c_max,
		constraints);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	massVectorMultpilyKern << <jacobian_bound_blocks, threads >> > (
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

	matrixMulKern << <bl, th >> > (jacobian, jacobian_transposed, A, nConstraints, 3 * nParticles);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	jaccobi(nConstraints, A, b, lambda, new_lambda, c_min, c_max, 50);

	applyForce << <particle_bound_blocks, threads >> > (new_lambda, jacobian_transposed, dx, dy, dz, nParticles, nConstraints, dt);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

ConstraintSolver::ConstraintSolver(int particles) : nParticles{ particles }
{
	// set pointers to 0 - this way it will be easy to distinguish whether they have already been allocated or not
	dev_jacobian = 0;
	dev_jacobian_transposed = 0;
	dev_A = 0;
	dev_b = 0;
	dev_lambda = 0;
	dev_new_lambda = 0;
	dev_c_min = 0;
	dev_c_max = 0;

	ConstraintStorage::Instance.initInstance();
}

ConstraintSolver::~ConstraintSolver()
{
	gpuErrchk(cudaFree(dev_jacobian));
	gpuErrchk(cudaFree(dev_jacobian_transposed));
	gpuErrchk(cudaFree(dev_A));
	gpuErrchk(cudaFree(dev_b));
	gpuErrchk(cudaFree(dev_lambda));
	gpuErrchk(cudaFree(dev_new_lambda));
	gpuErrchk(cudaFree(dev_c_min));
	gpuErrchk(cudaFree(dev_c_max));
}

void ConstraintSolver::calculateForces(
	float* x, float* y, float* z,
	float* new_x, float* new_y, float* new_z,
	float* dx, float* dy, float* dz,
	float* vx, float* vy, float* vz,
	float* invmass, float dt
)
{
	gpuErrchk(cudaMemset(dx, 0, nParticles * sizeof(float)));
	gpuErrchk(cudaMemset(dy, 0, nParticles * sizeof(float)));
	gpuErrchk(cudaMemset(dz, 0, nParticles * sizeof(float)));

	thrust::device_ptr<float> dx_ptr = thrust::device_pointer_cast(dx);
	thrust::device_ptr<float> dy_ptr = thrust::device_pointer_cast(dy);
	thrust::device_ptr<float> dz_ptr = thrust::device_pointer_cast(dz);

	thrust::device_ptr<float> x_ptr = thrust::device_pointer_cast(x);
	thrust::device_ptr<float> y_ptr = thrust::device_pointer_cast(y);
	thrust::device_ptr<float> z_ptr = thrust::device_pointer_cast(z);

	thrust::device_ptr<float> nx_ptr = thrust::device_pointer_cast(new_x);
	thrust::device_ptr<float> ny_ptr = thrust::device_pointer_cast(new_y);
	thrust::device_ptr<float> nz_ptr = thrust::device_pointer_cast(new_z);

	//this->projectConstraints<SurfaceConstraint>(invmass, new_x, new_y, new_z, dx, dy, dz, vx, vy, vz, dt, ConstraintType::SURFACE, true);
	//this->projectConstraints<SurfaceConstraint>(invmass, new_x, new_y, new_z, dx, dy, dz, vx, vy, vz, dt, ConstraintType::SURFACE, false);
	//this->projectConstraints<DistanceConstraint>(invmass, new_x, new_y, new_z, dx, dy, dz, vx, vy, vz, dt, ConstraintType::DISTANCE, true);
	//this->projectConstraints<DistanceConstraint>(invmass, new_x, new_y, new_z, dx, dy, dz, vx, vy, vz, dt, ConstraintType::DISTANCE, false);

	//thrust::transform(x_ptr, x_ptr + nParticles, dx_ptr, x_ptr, thrust::plus<float>());
	//thrust::transform(y_ptr, y_ptr + nParticles, dy_ptr, y_ptr, thrust::plus<float>());
	//thrust::transform(z_ptr, z_ptr + nParticles, dz_ptr, z_ptr, thrust::plus<float>());
	//thrust::transform(nx_ptr, nx_ptr + nParticles, dx_ptr, nx_ptr, thrust::plus<float>());
	//thrust::transform(ny_ptr, ny_ptr + nParticles, dy_ptr, ny_ptr, thrust::plus<float>());
	//thrust::transform(nz_ptr, nz_ptr + nParticles, dz_ptr, nz_ptr, thrust::plus<float>());

	gpuErrchk(cudaMemset(dx, 0, nParticles * sizeof(float)));
	gpuErrchk(cudaMemset(dy, 0, nParticles * sizeof(float)));
	gpuErrchk(cudaMemset(dz, 0, nParticles * sizeof(float)));

	//this->projectConstraints<SurfaceConstraint>(invmass, new_x, new_y, new_z, dx, dy, dz, vx, vy, vz, dt, ConstraintType::SURFACE, true);
	//this->projectConstraints<SurfaceConstraint>(invmass, new_x, new_y, new_z, dx, dy, dz, vx, vy, vz, dt, ConstraintType::SURFACE, false);
	//this->projectConstraints<DistanceConstraint>(invmass, new_x, new_y, new_z, dx, dy, dz, vx, vy, vz, dt, ConstraintType::DISTANCE, true);
	//this->projectConstraints<DistanceConstraint>(invmass, new_x, new_y, new_z, dx, dy, dz, vx, vy, vz, dt, ConstraintType::DISTANCE, false);

	this->projectConstraints<SurfaceConstraint>(invmass, x, y, z, dx, dy, dz, vx, vy, vz, dt, ConstraintType::SURFACE, true);
	this->projectConstraints<SurfaceConstraint>(invmass, x, y, z, dx, dy, dz, vx, vy, vz, dt, ConstraintType::SURFACE, false);
	this->projectConstraints<DistanceConstraint>(invmass, x, y, z, dx, dy, dz, vx, vy, vz, dt, ConstraintType::DISTANCE, true);
	this->projectConstraints<DistanceConstraint>(invmass, x, y, z, dx, dy, dz, vx, vy, vz, dt, ConstraintType::DISTANCE, false);

	thrust::transform(nx_ptr, nx_ptr + nParticles, dx_ptr, nx_ptr, thrust::plus<float>());
	thrust::transform(ny_ptr, ny_ptr + nParticles, dy_ptr, ny_ptr, thrust::plus<float>());
	thrust::transform(nz_ptr, nz_ptr + nParticles, dz_ptr, nz_ptr, thrust::plus<float>());
	
	ConstraintStorage::Instance.clearConstraints();
}

void ConstraintSolver::setStaticConstraints(std::vector<std::pair<int, int>> pairs, float d)
{
	std::vector<DistanceConstraint> cpu_constraints;
	for (const auto& pair : pairs)
	{
		cpu_constraints.push_back(DistanceConstraint().init(d, pair.first, pair.second, ConstraintLimitType::EQ));
	}

	ConstraintStorage::Instance.setStaticConstraints<DistanceConstraint>(cpu_constraints.data(), cpu_constraints.size(), ConstraintType::DISTANCE);
}

void ConstraintSolver::addDynamicConstraints(List* collisions, int* sums, float d, ConstraintLimitType type)
{
	ConstraintStorage::Instance.addCollisions(collisions, sums, type, d, nParticles);
}

void ConstraintSolver::addSurfaceConstraints(SurfaceConstraint* surfaceConstraints, int nSurfaceConstraints)
{
	ConstraintStorage::Instance.setDynamicConstraints<SurfaceConstraint>(surfaceConstraints, nSurfaceConstraints, ConstraintType::SURFACE);
}

void ConstraintSolver::allocateArrays(int nConstraints)
{
	if (nConstraints > nConstraintsMaxAllocated)
	{
		if (dev_jacobian != 0)
			gpuErrchk(cudaFree(dev_jacobian));
		gpuErrchk(cudaMalloc((void**)&dev_jacobian, 3 * nParticles * nConstraints * sizeof(float)));
		gpuErrchk(cudaMemset(dev_jacobian, 0, 3 * nParticles * nConstraints * sizeof(float)));

		if (dev_jacobian_transposed != 0)
			gpuErrchk(cudaFree(dev_jacobian_transposed));
		gpuErrchk(cudaMalloc((void**)&dev_jacobian_transposed, 3 * nParticles * nConstraints * sizeof(float)));
		gpuErrchk(cudaMemset(dev_jacobian_transposed, 0, 3 * nParticles * nConstraints * sizeof(float)));

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
	else this->clearArrays(nConstraints);
}

void ConstraintSolver::clearArrays(int nConstraints)
{
	gpuErrchk(cudaMemset(dev_jacobian, 0, 3 * nParticles * nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_jacobian_transposed, 0, 3 * nParticles * nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_b, 0, nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_new_lambda, 0, nConstraints * sizeof(float)));
}
