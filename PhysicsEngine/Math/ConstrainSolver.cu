#include "ConstrainSolver.cuh"
#include "../GpuErrorHandling.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/for_each.h>
#include <thrust/device_ptr.h>
#include "LinearSolver.cuh"

#define SHMEM_SIZE 1024

__global__ void fillJacobiansKern(
	int constrainsAmount, int particles,
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz,
	float* jacobian, float* velocity_jacobian,
	DistanceConstrain* constrains)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= constrainsAmount) return;
	for (int i = 0; i < constrains[index].n; i++)
	{
		constrains[index].positionDerivative(x, y, z, vx, vy, vz, i, &jacobian[index * 3 * particles + 3 * constrains[index].dev_indexes[i]]);
		constrains[index].timePositionDerivative(x, y, z, vx, vy, vz, i, &velocity_jacobian[index * 3 * particles + 3 * constrains[index].dev_indexes[i]]);
	}
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


		if (row < N && col < N)
		{
			if (i + threadIdx.x < K)
				s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i + threadIdx.x];
			if (i + threadIdx.y < K)
				s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];
		}
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

ConstrainSolver::ConstrainSolver(int particles, int constrainsNumber) : particles{ particles }, constrainsNumber{constrainsNumber}
{
	int* indexes = new int[2];
	indexes[0] = 0;
	indexes[1] = 1;
	DistanceConstrain* constrains = new DistanceConstrain{ 0.1f, indexes };
	gpuErrchk(cudaMalloc((void**)&dev_constrains, constrainsNumber * sizeof(DistanceConstrain)));
	gpuErrchk(cudaMemcpy(dev_constrains, constrains, constrainsNumber * sizeof(DistanceConstrain), cudaMemcpyHostToDevice));
	delete[] indexes;

	gpuErrchk(cudaMalloc((void**)&dev_jacobian, 3 * particles * constrainsNumber * sizeof(float)));
	gpuErrchk(cudaMemset(dev_jacobian, 0, 3 * particles * constrainsNumber * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_jacobian_transposed, 3 * particles * constrainsNumber * sizeof(float)));
	gpuErrchk(cudaMemset(dev_jacobian_transposed, 0, 3 * particles * constrainsNumber * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_velocity_jacobian, 3 * particles * constrainsNumber * sizeof(float)));
	gpuErrchk(cudaMemset(dev_velocity_jacobian, 0, 3 * particles * constrainsNumber * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_A, 3 * particles * 3 * particles * sizeof(float)));
	gpuErrchk(cudaMemset(dev_A, 0, 3 * particles * 3 * particles * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_b, 3 * particles * sizeof(float)));
	gpuErrchk(cudaMemset(dev_b, 0, 3 * particles * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_lambda, 3 * particles * sizeof(float)));
	gpuErrchk(cudaMemset(dev_lambda, 0, 3 * particles * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_new_lambda, 3 * particles * sizeof(float)));
	gpuErrchk(cudaMemset(dev_new_lambda, 0, 3 * particles * sizeof(float)));

}

ConstrainSolver::~ConstrainSolver()
{
	gpuErrchk(cudaFree(dev_constrains));
	gpuErrchk(cudaFree(dev_jacobian));
	gpuErrchk(cudaFree(dev_jacobian_transposed));
	gpuErrchk(cudaFree(dev_velocity_jacobian));
	gpuErrchk(cudaFree(dev_A));
	gpuErrchk(cudaFree(dev_b));
	gpuErrchk(cudaFree(dev_lambda));
	gpuErrchk(cudaFree(dev_new_lambda));
}

void ConstrainSolver::calculateForces(
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz,
	float* invmass
)
{
	unsigned int threads = 32;
	int blocks = ceilf(constrainsNumber / (float)threads);

	fillJacobiansKern << < blocks, threads >> > (constrainsNumber, particles,
		x, y, z,
		vx, vy, vz,
		dev_jacobian, dev_velocity_jacobian,
		dev_constrains);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	transposeKern << <blocks, threads>> > (
		3 * particles,
		constrainsNumber,
		dev_jacobian,
		dev_jacobian_transposed);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	massVectorMultpilyKern << <blocks, threads >> > (
		3 * particles,
		constrainsNumber,
		invmass,
		dev_jacobian);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	unsigned int BLOCKS_X = (3 * particles + threads - 1) / threads;
	unsigned int BLOCKS_Y = (3 * particles + threads - 1) / threads;

	dim3 t{ threads, threads };
	dim3 b{ BLOCKS_X, BLOCKS_Y };

	matrixMulKern<<<b, t>>>(dev_jacobian, dev_jacobian_transposed, dev_A, 3 * particles, constrainsNumber);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	jaccobiKern << <blocks, threads >> > (3 * particles, dev_A, dev_b, dev_lambda, dev_new_lambda);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//std::swap(dev_lambda, dev_new_lambda);

	int N = particles * 3 * particles * 3;
	float* tmp = new float[N];
	cudaMemcpy(tmp, dev_A, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 3 * particles; i++)
	{
		for (int j = 0; j < 3 * particles; j++)
		{
			std::cout << tmp[i * 3 * particles + j] << " ";
		}
		std::cout << std::endl;
	}
	/*float* tmp = new float[3 * particles * constrainsNumber];
	gpuErrchk(cudaMemcpy(tmp, dev_jacobian, 3 * particles * constrainsNumber * sizeof(float), cudaMemcpyDeviceToHost));
	for (int j = 0; j < constrainsNumber; j++)
	{
		for (int i = 0; i < particles; i++)
		{
			std::cout << " x: " << tmp[3 * i + j * constrainsNumber] << " y: " << tmp[3 * i + 1 + j * constrainsNumber] << " z: " << tmp[3 * i + 2 + j * constrainsNumber];
		}
		std::cout << "\n";
	}*/

}
