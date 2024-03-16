#pragma once
#include <vector>
#include <cuda_runtime.h>

struct Surface
{
	float a, b, c, d;
	float normal[3];
	Surface(float a, float b, float c, float d)
	{
		this->a = a;
		this->b = b;
		this->c = c;
		this->d = d;

		float len = sqrtf(a * a + b * b + c * c);
		normal[0] = a / len;
		normal[1] = b / len;
		normal[2] = c / len;
	}
};

class SurfaceCollisionFinder
{
	Surface* dev_surface;
	int* dev_hit;
	int nSurfaces;
public:
	SurfaceCollisionFinder(std::vector<Surface> surfaces, int nParticles);
	~SurfaceCollisionFinder();

	std::pair<float*, float*> findAndUpdateCollisions(
		int nParticles,
		float* x, float* y, float* z,
		float* vx, float* vy, float* vz);
};

