#pragma once
#include "../Constraint.cuh"

class DistanceConstraint : public Constraint
{
public:
	int p[2];
	__host__ __device__ DistanceConstraint init(float d, int p1, int p2, ConstraintLimitType type, float compliance = 0.f);
	__host__ __device__ float operator()(float* x, float* y, float* z);
	__host__ __device__ void positionDerivative(float* x, float* y, float* z, float* jacobian, int nParticles, int index);
	__device__ void directSolve(float* x, float* y, float* z, float* dx, float* dy, float* dz, int* nConstraintsPerParticle);
private:
	float d;
};