#include "ConstrainSolver.cuh"
#include "../GpuErrorHandling.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/transform.h>
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
	b[index] = -constrains[index](x, y, z, vx, vy, vz) - constrains[index].timeDerivative(x, y, z, vx, vy, vz);
		//for (int j = 0; j < particles; j++)
		//{
		//	b[i] -= jacobian[i * 3 * particles + 3 * j] * vx[j] / dt;
		//	b[i] -= jacobian[i * 3 * particles + 3 * j + 1] * vy[j] / dt;
		//	b[i] -= jacobian[i * 3 * particles + 3 * j + 2] * vz[j] / dt;
		//}
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

ConstrainSolver::ConstrainSolver(int particles) : nParticles{ particles }
{

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
	float* invmass, float* fc, float dt
)
{
	int N = nParticles * 3;
	float* tmp = new float[N * N];


	unsigned int threads = 32;


	// kernels bound by number of constraints
	int constraint_bound_blocks = (nConstraints + threads - 1) / threads;

	// kernels bound by the size of Jacobian
	int particle_bound_blocks = ((3 * nParticles * nConstraints) + threads - 1) / threads;

	fillJacobiansKern << < constraint_bound_blocks, threads >> > (nConstraints, nParticles,
		x, y, z,
		vx, vy, vz,
		dev_jacobian, dev_velocity_jacobian,
		dev_constrains);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	/*cudaMemcpy(tmp, dev_jacobian, N * nConstraints * sizeof(float), cudaMemcpyDeviceToHost);
	for (int k = 0; k < nConstraints; k++)
	{
		for (int i = 0; i < N; i++)
			std::cout << tmp[k * N + i] << " ";
		std::cout << "\n";

	}
	std::cout << "\n\n";*/

	
	fillResultVectorKern<<<constraint_bound_blocks, threads>>>(nParticles, nConstraints, dev_b,
		x, y, z,
		vx, vy, vz, dev_jacobian, dt,
		dev_constrains);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	/*cudaMemcpy(tmp, dev_b, nConstraints * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < nConstraints; i++)
			std::cout << tmp[i] << " ";
		std::cout << "\n";*/

	transposeKern << <particle_bound_blocks, threads>> > (
		3 * nParticles,
		nConstraints,
		dev_jacobian,
		dev_jacobian_transposed);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize())

	//cudaMemcpy(tmp, dev_jacobian_transposed, N * nConstraints * sizeof(float), cudaMemcpyDeviceToHost);
	//for (int k = 0; k < N; k++)
	//{
	//	for (int i = 0; i < nConstraints; i++)
	//		std::cout << tmp[k * nConstraints + i] << " ";
	//	std::cout << "\n";

	//}
	//std::cout << "\n\n";

		
	massVectorMultpilyKern << <particle_bound_blocks, threads >> > (
		3 * nParticles,
		nConstraints,
		invmass,
		dev_jacobian);

	//gpuErrchk(cudaGetLastError());
	//gpuErrchk(cudaDeviceSynchronize());

	/*cudaMemcpy(tmp, dev_jacobian, N * constrainsNumber * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++)
		std::cout << tmp[i] << " ";
	std::cout << "\n";*/

	unsigned int BLOCKS_X = (nConstraints + threads - 1) / threads;
	unsigned int BLOCKS_Y = (nConstraints + threads - 1) / threads;

	dim3 t{ threads, threads };
	dim3 b{ BLOCKS_X, BLOCKS_Y };

	matrixMulKern<<<b, t>>>(dev_jacobian, dev_jacobian_transposed, dev_A, nConstraints, 3 * nParticles);

	//cudaMemcpy(tmp, dev_A, nConstraints *  nConstraints * sizeof(float), cudaMemcpyDeviceToHost);
	////cudaMemcpy(tmp + 1, dev_b, constrainsNumber * sizeof(float), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < nConstraints; i++)
	//{
	//	for(int j = 0; j < nConstraints; j++)
	//		std::cout << tmp[i * nConstraints + j] << " ";
	//	std::cout << "\n";

	//}
	//std::cout << "\n\n";


	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	jaccobiKern << <constraint_bound_blocks, threads >> > (nConstraints, dev_A, dev_b, dev_lambda, dev_new_lambda);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//std::swap(dev_lambda, dev_new_lambda);
	applyForce << <particle_bound_blocks, threads >> > (dev_new_lambda, dev_jacobian_transposed, fc, 3 * nParticles, nConstraints);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	/*cudaMemcpy(tmp, dev_new_lambda, constrainsNumber* sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < constrainsNumber; i++)
	{
		std::cout << tmp[i] << " ";
	}
	std::cout << "\n\n";*/



	/*cudaMemcpy(tmp, fc, N * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++)
	{
		std::cout << tmp[i] << " ";
	}
	std::cout << "\n\n";*/
}

void ConstrainSolver::setConstraints(std::vector<std::pair<int, int>> pairs, float d)
{
	nConstraints = pairs.size();
	for (const auto& pair : pairs)
	{
		int tmp[2];
		tmp[0] = pair.first;
		tmp[1] = pair.second;
		cpu_constraints.push_back(DistanceConstrain(d, tmp));
	}
	gpuErrchk(cudaMalloc(&dev_constrains, sizeof(DistanceConstrain) * nConstraints));
	gpuErrchk(cudaMemcpy(dev_constrains, cpu_constraints.data(), cpu_constraints.size() * sizeof(DistanceConstrain), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc((void**)&dev_jacobian, 3 * nParticles * nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_jacobian, 0, 3 * nParticles * nConstraints * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_jacobian_transposed, 3 * nParticles * nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_jacobian_transposed, 0, 3 * nParticles * nConstraints * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_velocity_jacobian, 3 * nParticles * nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_velocity_jacobian, 0, 3 * nParticles * nConstraints * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_A, nConstraints * nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_A, 0, nConstraints * nConstraints * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_b, nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_b, 0, nConstraints * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_lambda, nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_lambda, 0, nConstraints * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_new_lambda, nConstraints * sizeof(float)));
	gpuErrchk(cudaMemset(dev_new_lambda, 0, nConstraints * sizeof(float)));

}
