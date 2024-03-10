#include "ConstrainSolver.cuh"
#include "../GpuErrorHandling.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "LinearSolver.cuh"
#include "../Constants.hpp"

#define SHMEM_SIZE 1024

#define AxisIndex(x) (x - MINDIMENSION) / PARTICLERADIUS
#define PositionToGrid(x, y, z)  AxisIndex(x) + CUBESPERDIMENSION * (AxisIndex(y) + CUBESPERDIMENSION * AxisIndex(z))
#define DistanceSquared(x1, y1, z1, x2, y2, z2) (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2)

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

__global__ void findCollisions(
	float* x, float* y, float* z,
	unsigned int* mapping, unsigned int* grid,
	int* gridCubeStart, int* gridCubeEnd, int nParticles)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;

	unsigned int xIdx = AxisIndex(x[index]);
	unsigned int yIdx = AxisIndex(y[index]);
	unsigned int zIdx = AxisIndex(z[index]);

	unsigned int minX = min(xIdx - 1, 0);
	unsigned int minY = min(yIdx - 1, 0);
	unsigned int minZ = min(zIdx - 1, 0);
	unsigned int maxX = max(xIdx + 1, (int)(CUBESPERDIMENSION - 1));
	unsigned int maxY = max(yIdx + 1, (int)(CUBESPERDIMENSION - 1));
	unsigned int maxZ = max(zIdx + 1, (int)(CUBESPERDIMENSION - 1));

	for (int i = minX; i < maxX; i++)
	{
		for (int j = minY; j < maxY; j++)
		{
			for (int k = minZ; k < maxZ; k++)
			{
				int currentCube = i + CUBESPERDIMENSION * (j + CUBESPERDIMENSION * k);	
				int first = gridCubeStart[currentCube];
				int last = gridCubeEnd[currentCube];

				if (first == -1) continue;
				for (int it = first; it <= last; it++)
				{
					int particle = mapping[it];
					float px = x[particle];
					float py = y[particle];
					float pz = z[particle];

					float distanceSq = DistanceSquared(x[index], y[index], z[index], px, py, pz);
					if (distanceSq < PARTICLERADIUS)
					{
						// tutaj dodaj co siê ma wydarzyæ w wypadku gdy siê zderzy³y
					}
				}
			}
		}
	}
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

__global__ void generateGridIndiciesKern(float* x, float* y, float* z, unsigned int* indicies, unsigned int* mapping, int nParticles)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;
	indicies[index] = PositionToGrid(x[index], y[index], z[index]);
	mapping[index] = index;
}

__global__ void identifyGridCubeStartEndKern(unsigned int* grid, int* grid_cube_start, int* grid_cube_end, int nParticles)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;

	int gridIndex = grid[index];
	if (index == 0) {
		grid_cube_start[gridIndex] = 0;
		return;
	}
	if (index == nParticles - 1)
	{
		grid_cube_end[gridIndex] = nParticles - 1;
	}

	const unsigned int prevGridIndex = grid[index - 1];
	if (gridIndex != prevGridIndex)
	{
		grid_cube_end[prevGridIndex] = index - 1;
		grid_cube_start[prevGridIndex] = index;
		if (index == nParticles - 1)
		{
			grid_cube_end[gridIndex] = index;
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
	gpuErrchk(cudaFree(dev_grid_cube_start));
	gpuErrchk(cudaFree(dev_grid_cube_end));
	gpuErrchk(cudaFree(dev_grid_index));
	gpuErrchk(cudaFree(dev_mapping));

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

	generateGridIndiciesKern << <particle_bound_blocks, threads >> > (x, y, z, dev_grid_index, dev_mapping, nParticles);
	
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::sort_by_key(thrust_grid, thrust_grid + nParticles, thrust_mapping);
	thrust::fill(thrust_grid_cube_start, thrust_grid_cube_start + TOTALCUBES, -1);
	thrust::fill(thrust_grid_cube_end, thrust_grid_cube_end + TOTALCUBES, -1);

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

	gpuErrchk(cudaMalloc((void**)&dev_grid_index, nParticles * sizeof(unsigned int)));
	
	gpuErrchk(cudaMalloc((void**)&dev_mapping, nParticles * sizeof(unsigned int)));

	gpuErrchk(cudaMalloc((void**)&dev_grid_cube_start, TOTALCUBES * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_grid_cube_end, TOTALCUBES * sizeof(unsigned int)));

	thrust_grid = thrust::device_pointer_cast<unsigned int>(dev_grid_index);
	thrust_mapping = thrust::device_pointer_cast<unsigned int>(dev_mapping);
	thrust_grid_cube_start = thrust::device_pointer_cast<int>(dev_grid_cube_start);
	thrust_grid_cube_end = thrust::device_pointer_cast<int>(dev_grid_cube_end);
}
