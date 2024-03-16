#pragma once
#include <thrust/device_ptr.h>
#include "../List/List.cuh"

class CollisionGrid 
{
	unsigned int* dev_grid_index;
	unsigned int* dev_mapping;
	int* dev_grid_cube_start;
	int* dev_grid_cube_end;

	thrust::device_ptr<unsigned int> thrust_grid;
	thrust::device_ptr<unsigned int> thrust_mapping;
	thrust::device_ptr<int> thrust_grid_cube_start;
	thrust::device_ptr<int> thrust_grid_cube_end;

	List* dev_collision_lists;

public:
	CollisionGrid(int nParticles);
	~CollisionGrid();
	void findCollisions(float* x, float* y, float* z, int nParticles);
};

