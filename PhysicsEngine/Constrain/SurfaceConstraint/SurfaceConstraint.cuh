#pragma once
#include "../Constrain.cuh"
#include "../../Collision/Surface.cuh"

#include <cuda_runtime.h>

class SurfaceConstraint	: public Constrain
{
public:
	__host__ __device__ SurfaceConstraint init(float d, int particle, Surface s);

	__host__ __device__ float operator()(float* x, float* y, float* z,
		float* vx, float* vy, float* vz);
	__host__ __device__ void positionDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz, int index, float* output);

	int p[1];
private:
	float r;
	Surface s;
};

