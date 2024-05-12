#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "../Constraint/SurfaceConstraint/SurfaceConstraint.cuh"
#include "Surface.cuh"


class SurfaceCollisionFinder
{
	Surface* dev_surface;
	SurfaceConstraint* dev_foundCollisions;
	int* dev_hit, *dev_hit_sign;
	int nSurfaces, nConstraintsMaxAllocated;
public:
	SurfaceCollisionFinder(std::vector<Surface> surfaces, int nParticles);
	~SurfaceCollisionFinder();

	void setSurfaces(std::vector<Surface> surfaces, int nParticles);

	void findAndUpdateCollisions(
		int nParticles,
		float* x, float* y, float* z);
};

