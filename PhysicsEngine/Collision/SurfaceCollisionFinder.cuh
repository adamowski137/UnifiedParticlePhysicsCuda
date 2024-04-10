#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "../Constraint/SurfaceConstraint/SurfaceConstraint.cuh"
#include "Surface.cuh"


class SurfaceCollisionFinder
{
	Surface* dev_surface;
	int* dev_hit, *dev_hit_sign;
	int nSurfaces;
public:
	SurfaceCollisionFinder(std::vector<Surface> surfaces, int nParticles);
	~SurfaceCollisionFinder();

	void setSurfaces(std::vector<Surface> surfaces, int nParticles);

	std::pair<SurfaceConstraint*, int> findAndUpdateCollisions(
		int nParticles,
		float* x, float* y, float* z,
		float* new_x, float* new_y, float* new_z);
};

