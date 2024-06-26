#include "LinearSystemConstraintSolver.cuh"
#include "../../../GpuErrorHandling.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "../../../Constants.hpp"

#define SHMEM_SIZE 1024


template<typename T>
__global__ void fillJacobiansKern(
	int nConstraints, int nParticles,
	float* x, float* y, float* z,
	float* jacobian,
	T* constrains)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nConstraints) return;
	constrains[index].positionDerivative(x, y, z, jacobian, nParticles, index);
}


template <typename T>
__global__ void fillResultVectorKern(int particles, int constrainsNumber, float* b,
	float* x, float* y, float* z,
	float* jacobian,
	float* dev_c_min, float* dev_c_max,
	T* constrains)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= constrainsNumber) return;
	b[index] = -(constrains[index])(x, y, z);
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


__global__ void applyForce(float* new_lambda, float* jacobi_transposed, float* dx, float* dy, float* dz, int* mode, float dt, int nParticles, int nConstraints)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < nParticles)
	{
		for (int i = 0; i < nConstraints; i++)
		{
			dx[index] += new_lambda[i] * jacobi_transposed[(3 * index + 0) * nConstraints + i] * dt;
			dy[index] += new_lambda[i] * jacobi_transposed[(3 * index + 1) * nConstraints + i] * dt;
			dz[index] += new_lambda[i] * jacobi_transposed[(3 * index + 2) * nConstraints + i] * dt;
		}
	}
}

template<typename T>
void fillJacobiansWrapper(int nConstraints, int nParticles,
	float* x, float* y, float* z, int* mode,
	float* dx, float* dy, float* dz,
	float* jacobian,
	float* jacobian_transposed, float* A,
	float* b, float dt,
	float* invmass, float* lambda, float* new_lambda, float* c_min, float* c_max,
	T* constraints, int iterations)
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
		jacobian,
		constraints);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillResultVectorKern << <constraint_bound_blocks, threads >> > (nParticles, nConstraints, b,
		x, y, z,
		jacobian,
		c_min, c_max,
		constraints);

	/*thrust::device_ptr<float> b_ptr = thrust::device_pointer_cast(b);
	std::cout << b_ptr[10] << "\n";*/

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	massVectorMultpilyKern << <jacobian_bound_blocks, threads >> > (
		3 * nParticles,
		nConstraints,
		invmass,
		jacobian);


	transposeKern << <jacobian_bound_blocks, threads >> > (
		3 * nParticles,
		nConstraints,
		jacobian,
		jacobian_transposed);



	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	unsigned int BLOCKS_X = (nConstraints + threads - 1) / threads;
	unsigned int BLOCKS_Y = (nConstraints + threads - 1) / threads;

	dim3 th{ threads, threads };
	dim3 bl{ BLOCKS_X, BLOCKS_Y };

	matrixMulKern << <bl, th >> > (jacobian, jacobian_transposed, A, nConstraints, 3 * nParticles);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//jaccobi(nConstraints, A, b, lambda, new_lambda, c_min, c_max, iterations);
	gauss_seidel_cpu(nConstraints, A, b, lambda, new_lambda, c_min, c_max, iterations);

	applyForce << <particle_bound_blocks, threads >> > (new_lambda, jacobian_transposed, dx, dy, dz, mode, dt, nParticles, nConstraints);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

LinearSystemConstraintSolver::LinearSystemConstraintSolver(int particles) : ConstraintSolver(particles)
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

	nConstraintsMaxAllocated = 1;
	this->allocateArrays(50);

}

LinearSystemConstraintSolver::~LinearSystemConstraintSolver()
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

void LinearSystemConstraintSolver::calculateForces(
	float* x, float* y, float* z, int* mode,
	float* new_x, float* new_y, float* new_z,
	float* invmass, float dt, int iterations
)
{
	int num_iterations = 1;
	for (int i = 0; i < num_iterations; i++)
	{
		gpuErrchk(cudaMemset(dev_dx, 0, nParticles * sizeof(float)));
		gpuErrchk(cudaMemset(dev_dy, 0, nParticles * sizeof(float)));
		gpuErrchk(cudaMemset(dev_dz, 0, nParticles * sizeof(float)));

		thrust::device_ptr<float> thrust_x(new_x);
		thrust::device_ptr<float> thrust_y(new_y);
		thrust::device_ptr<float> thrust_z(new_z);

		thrust::device_ptr<float> thrust_dx(dev_dx);
		thrust::device_ptr<float> thrust_dy(dev_dy);
		thrust::device_ptr<float> thrust_dz(dev_dz);

		this->projectConstraints<SurfaceConstraint>(invmass, new_x, new_y, new_z, mode, dt / num_iterations, true, iterations);
		this->projectConstraints<DistanceConstraint>(invmass, new_x, new_y, new_z, mode, dt / num_iterations, true, iterations);

		//this->projectConstraints<SurfaceConstraint>(invmass, new_x, new_y, new_z, mode, dt / num_iterations, false, iterations);
		//this->projectConstraints<DistanceConstraint>(invmass, new_x, new_y, new_z, mode, dt / num_iterations, false, iterations);

		auto rigidBodyConstraints = ConstraintStorage<RigidBodyConstraint>::Instance.getCpuConstraints();

		ConstraintArgsBuilder builder{};
		builder.initBase(new_x, new_y, new_z, dev_dx, dev_dy, dev_dz, invmass, nullptr, dt / num_iterations);

		for (int i = 0; i < rigidBodyConstraints.size(); i++)
		{
			rigidBodyConstraints[i]->calculateShapeCovariance(new_x, new_y, new_z, invmass);
			rigidBodyConstraints[i]->calculatePositionChange(builder.build());
		}

 		thrust::transform(thrust_x, thrust_x + nParticles, thrust_dx, thrust_x, thrust::plus<float>());
		thrust::transform(thrust_y, thrust_y + nParticles, thrust_dy, thrust_y, thrust::plus<float>());
		thrust::transform(thrust_z, thrust_z + nParticles, thrust_dz, thrust_z, thrust::plus<float>());
	}

	this->clearAllConstraints();
}

void LinearSystemConstraintSolver::calculateStabilisationForces(
	float* x, float* y, float* z, int* mode,
	float* new_x, float* new_y, float* new_z,
	float* invmass, float dt, int iterations)
{
	gpuErrchk(cudaMemset(dev_dx, 0, nParticles * sizeof(float)));
	gpuErrchk(cudaMemset(dev_dy, 0, nParticles * sizeof(float)));
	gpuErrchk(cudaMemset(dev_dz, 0, nParticles * sizeof(float)));

	thrust::device_ptr<float> thrust_x(x);
	thrust::device_ptr<float> thrust_y(y);
	thrust::device_ptr<float> thrust_z(z);

	thrust::device_ptr<float> thrust_new_x(new_x);
	thrust::device_ptr<float> thrust_new_y(new_y);
	thrust::device_ptr<float> thrust_new_z(new_z);

	thrust::device_ptr<float> thrust_dx(dev_dx);
	thrust::device_ptr<float> thrust_dy(dev_dy);
	thrust::device_ptr<float> thrust_dz(dev_dz);

	this->projectConstraints<DistanceConstraint>(invmass, x, y, z, mode, dt, true, iterations);
	this->projectConstraints<SurfaceConstraint>(invmass, x, y, z, mode, dt, true, iterations);

	thrust::transform(thrust_new_x, thrust_new_x + nParticles, thrust_dx, thrust_new_x, thrust::plus<float>());
	thrust::transform(thrust_new_y, thrust_new_y + nParticles, thrust_dy, thrust_new_y, thrust::plus<float>());
	thrust::transform(thrust_new_z, thrust_new_z + nParticles, thrust_dz, thrust_new_z, thrust::plus<float>());

	thrust::transform(thrust_x, thrust_x + nParticles, thrust_dx, thrust_x, thrust::plus<float>());
	thrust::transform(thrust_y, thrust_y + nParticles, thrust_dy, thrust_y, thrust::plus<float>());
	thrust::transform(thrust_z, thrust_z + nParticles, thrust_dz, thrust_z, thrust::plus<float>());

	this->clearAllConstraints();
}

void LinearSystemConstraintSolver::allocateArrays(int nConstraints)
{
	if (nConstraints > nConstraintsMaxAllocated)
	{
		while (nConstraints > nConstraintsMaxAllocated)
		{
			nConstraintsMaxAllocated *= 2;
		}

		if (dev_jacobian != 0)
			gpuErrchk(cudaFree(dev_jacobian));
		gpuErrchk(cudaMalloc((void**)&dev_jacobian, 3 * nParticles * nConstraintsMaxAllocated * sizeof(float)));
		gpuErrchk(cudaMemset(dev_jacobian, 0, 3 * nParticles * nConstraintsMaxAllocated * sizeof(float)));

		if (dev_jacobian_transposed != 0)
			gpuErrchk(cudaFree(dev_jacobian_transposed));
		gpuErrchk(cudaMalloc((void**)&dev_jacobian_transposed, 3 * nParticles * nConstraintsMaxAllocated * sizeof(float)));
		gpuErrchk(cudaMemset(dev_jacobian_transposed, 0, 3 * nParticles * nConstraintsMaxAllocated * sizeof(float)));

		if (dev_A != 0)
			gpuErrchk(cudaFree(dev_A));
		gpuErrchk(cudaMalloc((void**)&dev_A, nConstraintsMaxAllocated * nConstraintsMaxAllocated * sizeof(float)));
		gpuErrchk(cudaMemset(dev_A, 0, nConstraintsMaxAllocated * nConstraintsMaxAllocated * sizeof(float)));

		if (dev_b != 0)
			gpuErrchk(cudaFree(dev_b));
		gpuErrchk(cudaMalloc((void**)&dev_b, nConstraintsMaxAllocated * sizeof(float)));
		gpuErrchk(cudaMemset(dev_b, 0, nConstraintsMaxAllocated * sizeof(float)));

		if (dev_lambda != 0)
			gpuErrchk(cudaFree(dev_lambda));
		gpuErrchk(cudaMalloc((void**)&dev_lambda, nConstraintsMaxAllocated * sizeof(float)));
		gpuErrchk(cudaMemset(dev_lambda, 0, nConstraintsMaxAllocated * sizeof(float)));

		if (dev_new_lambda != 0)
			gpuErrchk(cudaFree(dev_new_lambda));
		gpuErrchk(cudaMalloc((void**)&dev_new_lambda, nConstraintsMaxAllocated * sizeof(float)));
		gpuErrchk(cudaMemset(dev_new_lambda, 0, nConstraintsMaxAllocated * sizeof(float)));

		if (dev_c_min != 0)
			gpuErrchk(cudaFree(dev_c_min));
		gpuErrchk(cudaMalloc((void**)&dev_c_min, nConstraintsMaxAllocated * sizeof(float)));
		gpuErrchk(cudaMemset(dev_c_min, 0, nConstraintsMaxAllocated * sizeof(float)));

		if (dev_c_max != 0)
			gpuErrchk(cudaFree(dev_c_max));
		gpuErrchk(cudaMalloc((void**)&dev_c_max, nConstraintsMaxAllocated * sizeof(float)));
		gpuErrchk(cudaMemset(dev_c_max, 0, nConstraintsMaxAllocated * sizeof(float)));

	}
	else this->clearArrays(nConstraints);
}

void LinearSystemConstraintSolver::clearArrays(int nConstraints)
{
	gpuErrchk(cudaMemset(dev_jacobian, 0, 3 * nParticles * nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_jacobian_transposed, 0, 3 * nParticles * nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_b, 0, nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_new_lambda, 0, nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_lambda, 0, nConstraints * sizeof(float)));
}
