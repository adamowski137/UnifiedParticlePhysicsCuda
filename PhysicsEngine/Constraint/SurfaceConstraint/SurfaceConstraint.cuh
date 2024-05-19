#pragma once
#include "../Constraint.cuh"
#include "../ConstraintArgs.hpp"
#include "../../Collision/Surface.cuh"

class SurfaceConstraint	: public Constraint
{
public:
	__host__ __device__ SurfaceConstraint init(float d, int particle, Surface s);

	__host__ __device__ float operator()(float* x, float* y, float* z);
	__host__ __device__ void positionDerivative(float* x, float* y, float* z, float* jacobian, int nParticles, int index);
	__device__ void directSolve(ConstraintArgs args);
	__host__ void directSolve_cpu(float* x, float* y, float* z, float* invmass, float dt, float* delta_lambda, int idx);
	int p[1];
private:
	float r;
	Surface s;
};

