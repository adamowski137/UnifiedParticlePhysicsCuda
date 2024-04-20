#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "../Constraint/SurfaceConstraint/SurfaceConstraint.cuh"
#include "Surface.cuh"


class SurfaceCollisionFinder
{
	Surface* dev_surface;
	SurfaceConstraint* dev_foundCollisions;
	int* dev_hit;
	int nSurfaces, nConstraintsMaxAllocated;
public:
	SurfaceCollisionFinder(std::vector<Surface> surfaces, int nParticles);
	~SurfaceCollisionFinder();

	void setSurfaces(std::vector<Surface> surfaces, int nParticles);

	std::pair<SurfaceConstraint*, int> findAndUpdateCollisions(
		int nParticles,
		float* x, float* y, float* z);
};

