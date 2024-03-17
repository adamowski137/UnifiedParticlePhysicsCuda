#pragma once
#include "../Constrain.cuh"
#include "../../Collision/Surface.hpp"

#include <cuda_runtime.h>

class SurfaceConstraint	: public Constrain
{
public:
	__host__ __device__ SurfaceConstraint(float d, int particle, Surface s);

	__host__ __device__ float operator()(float* x, float* y, float* z,
		float* vx, float* vy, float* vz);
	__host__ __device__  float timeDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz);
	__host__ __device__ void positionDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz, float* output);
	__host__ __device__  void timePositionDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz, float* output);

	int p;
private:
	float r;
	Surface s;
};

