#pragma once
#include <cuda_runtime.h>
#include "../Constraint.cuh"
#include "../../Collision/Surface.cuh"

class SurfaceConstraint	: public Constraint
{
public:
	__host__ __device__ SurfaceConstraint init(float d, int particle, Surface s);

	__host__ __device__ float operator()(float* x, float* y, float* z);
	__host__ __device__ void positionDerivative(float* x, float* y, float* z, float* jacobian, int nParticles, int index);
	__host__ __device__ void directSolve(float* x, float* y, float* z);

	int p[1];
private:
	float r;
	Surface s;
};

