#include "CollisionGrid.cuh" 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../Constants.hpp"
#include "../GpuErrorHandling.hpp"
#include <thrust/sort.h>

#define AxisIndex(x) (x - MINDIMENSION) / (2 * PARTICLERADIUS)
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
	int* gridCubeStart, int* gridCubeEnd, int nParticles)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;

	unsigned int xIdx = AxisIndex(x[index]);
	unsigned int yIdx = AxisIndex(y[index]);
	unsigned int zIdx = AxisIndex(z[index]);

	unsigned int minX = xIdx;
	unsigned int minY = yIdx;
	unsigned int minZ = zIdx;
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
						if (i == xIdx && j == yIdx && k == zIdx && particle >= index) continue;
						collisionList[index].addNode(particle);
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

	thrust_grid = thrust::device_pointer_cast<unsigned int>(dev_grid_index);
	thrust_mapping = thrust::device_pointer_cast<unsigned int>(dev_mapping);
	thrust_grid_cube_start = thrust::device_pointer_cast<int>(dev_grid_cube_start);
	thrust_grid_cube_end = thrust::device_pointer_cast<int>(dev_grid_cube_end);

	gpuErrchk(cudaMalloc((void**)&dev_collision_lists, nParticles * sizeof(List)));

}

CollisionGrid::~CollisionGrid()
{
	gpuErrchk(cudaFree(dev_grid_cube_start));
	gpuErrchk(cudaFree(dev_grid_cube_end));
	gpuErrchk(cudaFree(dev_grid_index));
	gpuErrchk(cudaFree(dev_mapping));

	gpuErrchk(cudaFree(dev_collision_lists));
}

void CollisionGrid::findCollisions(float* x, float* y, float* z, int nParticles)
{
	int threads = 32;
	int grid_bound_blocks = (TOTALCUBES + threads - 1) / threads;
	int particle_bound_blocks = (nParticles + threads - 1) / threads;

	generateGridIndiciesKern << <particle_bound_blocks, threads >> > (x, y, z, dev_grid_index, dev_mapping, nParticles);
	
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::sort_by_key(thrust_grid, thrust_grid + nParticles, thrust_mapping);
	//thrust::fill(thrust_grid_cube_start, thrust_grid_cube_start + TOTALCUBES, -1);
	//thrust::fill(thrust_grid_cube_end, thrust_grid_cube_end + TOTALCUBES, -1);
	fillArrayKern << <grid_bound_blocks, threads >> > (dev_grid_cube_start, TOTALCUBES, -1);
	fillArrayKern << <grid_bound_blocks, threads >> > (dev_grid_cube_end, TOTALCUBES, -1);

	clearCollisionsKern << <particle_bound_blocks, threads >> > (dev_collision_lists, nParticles);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	findCollisionsKern<<<particle_bound_blocks, threads>>>(dev_collision_lists, x, y, z, dev_mapping, dev_grid_index, dev_grid_cube_start, dev_grid_cube_end, nParticles);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

}


