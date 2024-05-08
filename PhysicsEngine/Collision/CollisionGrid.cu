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
					if (distanceSq < 4 * PARTICLERADIUS * PARTICLERADIUS)
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

__global__ void addCollisionsKern(List* collisions, int* counts, DistanceConstraint* constraints, float d, int nParticles)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;
	Node* p = collisions[index].head;
	int constrainIndex = counts[index] - 1;

	while (p != NULL)
	{
		constraints[constrainIndex] = DistanceConstraint().init(d, index, p->value, ConstraintLimitType::GEQ, 0.001f);
		p = p->next;
		constrainIndex--;
	}
}

CollisionGrid::CollisionGrid(int nParticles)
{

	gpuErrchk(cudaMalloc((void**)&dev_grid_index, nParticles * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_mapping, nParticles * sizeof(unsigned int)));

	gpuErrchk(cudaMalloc((void**)&dev_grid_cube_start, TOTALCUBES * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_grid_cube_end, TOTALCUBES * sizeof(unsigned int)));

	gpuErrchk(cudaMalloc((void**)&dev_counts, nParticles * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_collisions, nParticles * sizeof(List)));

	thrust_grid = thrust::device_pointer_cast<unsigned int>(dev_grid_index);
	thrust_mapping = thrust::device_pointer_cast<unsigned int>(dev_mapping);
	thrust_grid_cube_start = thrust::device_pointer_cast<int>(dev_grid_cube_start);
	thrust_grid_cube_end = thrust::device_pointer_cast<int>(dev_grid_cube_end);


	nConstraintsMaxAllocated = 128;
	gpuErrchk(cudaMalloc((void**)&dev_foundCollisions, nConstraintsMaxAllocated * sizeof(DistanceConstraint)));

}

CollisionGrid::~CollisionGrid()
{
	gpuErrchk(cudaFree(dev_grid_cube_start));
	gpuErrchk(cudaFree(dev_grid_cube_end));
	gpuErrchk(cudaFree(dev_grid_index));
	gpuErrchk(cudaFree(dev_mapping));
	gpuErrchk(cudaFree(dev_counts));
	gpuErrchk(cudaFree(dev_collisions));

}

void CollisionGrid::findAndUpdateCollisions(float* x, float* y, float* z, int nParticles)
{

	int threads = 32;
	int grid_bound_blocks = (TOTALCUBES + threads - 1) / threads;
	int particle_bound_blocks = (nParticles + threads - 1) / threads;

	// 1.
	// CLEAR COLLISIONS AND SET UP GRID

	clearCollisionsKern << <particle_bound_blocks, threads >> > (dev_collisions, nParticles);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemset(dev_counts, 0, sizeof(int) * nParticles));

	generateGridIndiciesKern << <particle_bound_blocks, threads >> > (x, y, z, dev_grid_index, dev_mapping, nParticles);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::sort_by_key(thrust_grid, thrust_grid + nParticles, thrust_mapping);

	fillArrayKern << <grid_bound_blocks, threads >> > (dev_grid_cube_start, TOTALCUBES, -1);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillArrayKern << <grid_bound_blocks, threads >> > (dev_grid_cube_end, TOTALCUBES, -1);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	identifyGridCubeStartEndKern << <particle_bound_blocks, threads >> > (dev_grid_index, dev_grid_cube_start, dev_grid_cube_end, nParticles);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// 2.
	// FIND COLLISIONS IN THE GRID

	findCollisionsKern << <particle_bound_blocks, threads >> > (dev_collisions, x, y, z, dev_mapping, dev_grid_index, dev_grid_cube_start, dev_grid_cube_end, nParticles, dev_counts);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//gpuErrchk(cudaMemset(sums, 0, nParticles * sizeof(int)));
	//
	//thrust::device_ptr<int> prefixSum{ sums };
	thrust::device_ptr<int> p = thrust::device_pointer_cast<int>(dev_counts);
	thrust::inclusive_scan(p, p + nParticles, p);

	// 3.
	// CREATE COLLISION CONSTRAINTS

	int nCollisions = p[nParticles - 1];
	if (nCollisions > nConstraintsMaxAllocated)
	{
		while (nCollisions > nConstraintsMaxAllocated)
			nConstraintsMaxAllocated *= 2;

		gpuErrchk(cudaFree(dev_foundCollisions));
		gpuErrchk(cudaMalloc((void**)&dev_foundCollisions, sizeof(DistanceConstraint) * nConstraintsMaxAllocated));
	}

	addCollisionsKern << <particle_bound_blocks, threads >> > (dev_collisions, dev_counts, dev_foundCollisions, 2 * PARTICLERADIUS, nParticles);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	if (nCollisions > 0)
		ConstraintStorage<DistanceConstraint>::Instance.addDynamicConstraints(dev_foundCollisions, nCollisions);
}


