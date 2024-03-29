#include "CollisionGrid.cuh" 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../Constants.hpp"
#include "../GpuErrorHandling.hpp"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <stdio.h>

#define AxisIndex(x) (int)((x - MINDIMENSION) / (CUBESIZE))
#define PositionToGrid(x, y, z)  AxisIndex(x) + CUBESPERDIMENSION * (AxisIndex(y) + CUBESPERDIMENSION * AxisIndex(z))
#define DistanceSquared(x1, y1, z1, x2, y2, z2) (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2)

__global__ void fillArrayKern(int* dst, int amount, int value)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= amount) return;

	dst[index] = value;
}

__global__ void findCollisionsKern(
	List* collisionList,
	float* x, float* y, float* z,
	unsigned int* mapping, unsigned int* grid,
	int* gridCubeStart, int* gridCubeEnd, int nParticles, int* collisionCount)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;

	// tak testowalem czy sumy prefiksowe dzialaja
	//collisionCount[index] = 1;
	unsigned int xIdx = AxisIndex(x[index]);
	unsigned int yIdx = AxisIndex(y[index]);
	unsigned int zIdx = AxisIndex(z[index]);

	unsigned int minX = xIdx;
	unsigned int minY = yIdx;
	unsigned int minZ = zIdx;
	unsigned int maxX = min(xIdx + 1, (int)(CUBESPERDIMENSION - 1));
	unsigned int maxY = min(yIdx + 1, (int)(CUBESPERDIMENSION - 1));
	unsigned int maxZ = min(zIdx + 1, (int)(CUBESPERDIMENSION - 1));

	for (int i = minX; i <= maxX; i++)
	{
		for (int j = minY; j <= maxY; j++)
		{
			for (int k = minZ; k <= maxZ; k++)
			{
				int currentCube = i + CUBESPERDIMENSION * (j + CUBESPERDIMENSION * k);
				int first = gridCubeStart[currentCube];
				int last = gridCubeEnd[currentCube];

				if (first == -1) continue;
				for (int it = first; it <= last; it++)
				{
					unsigned int particle = mapping[it];

					if (particle == index) continue;

					float px = x[particle];
					float py = y[particle];
					float pz = z[particle];

					float distanceSq = DistanceSquared(x[index], y[index], z[index], px, py, pz);
					if (distanceSq < PARTICLERADIUS * PARTICLERADIUS)
					{
						if (i == xIdx && j == yIdx && k == zIdx && particle < index) continue;
						collisionList[index].addNode(particle);
						collisionCount[index]++;
					}
				}
			}
		}
	}
}

__global__ void generateGridIndiciesKern(float* x, float* y, float* z, unsigned int* indicies, unsigned int* mapping, int nParticles)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;

	float xCur = x[index];
	float yCur = y[index];
	float zCur = z[index];

	if (x[index] >= MAXDIMENSION) xCur = MAXDIMENSION - 0.01f;
	if (y[index] >= MAXDIMENSION) yCur = MAXDIMENSION - 0.01f;
	if (z[index] >= MAXDIMENSION) zCur = MAXDIMENSION - 0.01f;

	if (x[index] <= MINDIMENSION) xCur = MINDIMENSION + 0.01f;
	if (y[index] <= MINDIMENSION) yCur = MINDIMENSION + 0.01f;
	if (z[index] <= MINDIMENSION) zCur = MINDIMENSION + 0.01f;

	indicies[index] = PositionToGrid(xCur, yCur, zCur);
	mapping[index] = index;
}

__global__ void identifyGridCubeStartEndKern(unsigned int* grid, int* grid_cube_start, int* grid_cube_end, int nParticles)
{
	const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= nParticles) return;

	const unsigned int current_grid_index = grid[index];

	if (index == 0) {
		grid_cube_start[current_grid_index] = 0;
		return;
	}
	if (index == nParticles - 1)
	{
		grid_cube_end[current_grid_index] = nParticles - 1;
	}

	const unsigned int prev_grid_index = grid[index - 1];
	if (current_grid_index != prev_grid_index) {
		grid_cube_end[prev_grid_index] = static_cast<int>(index - 1);
		grid_cube_start[current_grid_index] = index;

		if (index == nParticles - 1) { grid_cube_end[current_grid_index] = index; }
	}
}

__global__ void clearCollisionsKern(List* collisions, int nParticles)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;
	collisions[index].clearList();
}

CollisionGrid::CollisionGrid(int nParticles)
{

	gpuErrchk(cudaMalloc((void**)&dev_grid_index, nParticles * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_mapping, nParticles * sizeof(unsigned int)));

	gpuErrchk(cudaMalloc((void**)&dev_grid_cube_start, TOTALCUBES * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_grid_cube_end, TOTALCUBES * sizeof(unsigned int)));

	gpuErrchk(cudaMalloc((void**)&dev_counts, nParticles * sizeof(int)));

	thrust_grid = thrust::device_pointer_cast<unsigned int>(dev_grid_index);
	thrust_mapping = thrust::device_pointer_cast<unsigned int>(dev_mapping);
	thrust_grid_cube_start = thrust::device_pointer_cast<int>(dev_grid_cube_start);
	thrust_grid_cube_end = thrust::device_pointer_cast<int>(dev_grid_cube_end);

}

CollisionGrid::~CollisionGrid()
{
	gpuErrchk(cudaFree(dev_grid_cube_start));
	gpuErrchk(cudaFree(dev_grid_cube_end));
	gpuErrchk(cudaFree(dev_grid_index));
	gpuErrchk(cudaFree(dev_mapping));
	gpuErrchk(cudaFree(dev_counts));

}

void CollisionGrid::findCollisions(float* x, float* y, float* z, int nParticles, int* sums, List* collisions)
{
	int threads = 32;
	int grid_bound_blocks = (TOTALCUBES + threads - 1) / threads;
	int particle_bound_blocks = (nParticles + threads - 1) / threads;

	generateGridIndiciesKern << <particle_bound_blocks, threads >> > (x, y, z, dev_grid_index, dev_mapping, nParticles);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	thrust::sort_by_key(thrust_grid, thrust_grid + nParticles, thrust_mapping);

	//float* tmp_x = new float[nParticles];
	//float* tmp_y = new float[nParticles];
	//float* tmp_z = new float[nParticles];

	//cudaMemcpy(tmp_x, x, sizeof(float) * nParticles, cudaMemcpyDeviceToHost);
	//cudaMemcpy(tmp_y, y, sizeof(float) * nParticles, cudaMemcpyDeviceToHost);
	//cudaMemcpy(tmp_z, z, sizeof(float) * nParticles, cudaMemcpyDeviceToHost);

	//for (int i = 0; i < nParticles; i++)
	//{
	//	unsigned int index = thrust_mapping[i];
	//	std::cout << i << "  " << "particle index: " << index << " grid index: " << thrust_grid[i] << " | " 
	//		<< AxisIndex(tmp_x[index]) + CUBESPERDIMENSION * (AxisIndex(tmp_y[index]) + CUBESPERDIMENSION * AxisIndex(tmp_z[index])) << std::endl;
	//}

	//delete[] tmp_y;
	//delete[] tmp_z;
	//delete[] tmp_x;



	fillArrayKern << <grid_bound_blocks, threads >> > (dev_grid_cube_start, TOTALCUBES, -1);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillArrayKern << <grid_bound_blocks, threads >> > (dev_grid_cube_end, TOTALCUBES, -1);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	identifyGridCubeStartEndKern << <particle_bound_blocks, threads >> > (dev_grid_index, dev_grid_cube_start, dev_grid_cube_end, nParticles);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//for (int i = 0; i < TOTALCUBES; i++)
	//{
	//	if (thrust_grid_cube_start[i] == -1 || thrust_grid_cube_end[i] == -1) continue;
	//	std::cout << std::endl << i << " " << thrust_grid_cube_start[i] << " - " << thrust_grid_cube_end[i] << std::endl;
	//}

	clearCollisionsKern << <particle_bound_blocks, threads >> > (collisions, nParticles);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemset(dev_counts, 0, sizeof(int) * nParticles));

	findCollisionsKern << <particle_bound_blocks, threads >> > (collisions, x, y, z, dev_mapping, dev_grid_index, dev_grid_cube_start, dev_grid_cube_end, nParticles, dev_counts);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemset(sums, 0, nParticles * sizeof(int)));

	thrust::device_ptr<int> prefixSum{ sums };
	thrust::device_ptr<int> p = thrust::device_pointer_cast<int>(dev_counts);
	thrust::inclusive_scan(p, p + nParticles, prefixSum);
}


