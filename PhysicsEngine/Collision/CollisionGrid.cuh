#pragma once
#include <thrust/device_ptr.h>
#include "../List/List.cuh"
#include "../Constraint/ConstraintStorage.cuh"

class CollisionGrid 
{
	unsigned int* dev_grid_index;
	unsigned int* dev_mapping;
	int* dev_grid_cube_start;
	int* dev_grid_cube_end;
	int* dev_counts;

	List* dev_collisions;

	int nConstraintsMaxAllocated;
	DistanceConstraint* dev_foundCollisions;

	thrust::device_ptr<unsigned int> thrust_grid;
	thrust::device_ptr<unsigned int> thrust_mapping;
	thrust::device_ptr<int> thrust_grid_cube_start;
	thrust::device_ptr<int> thrust_grid_cube_end;

public:
	CollisionGrid(int nParticles);
	~CollisionGrid();
	void findAndUpdateCollisions(float* x, float* y, float* z, int* phase, int nParticles);
};

