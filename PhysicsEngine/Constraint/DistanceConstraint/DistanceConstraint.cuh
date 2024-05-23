#pragma once
#include "../Constraint.cuh"
#include "../ConstraintArgs.hpp"

class DistanceConstraint : public Constraint
{
public:
	int p[2];
	__host__ __device__ DistanceConstraint init(float d, int p1, int p2, ConstraintLimitType type, float k = 0.f, bool apply_friction = true);
	__host__ __device__ float operator()(float* x, float* y, float* z);
	__host__ __device__ void positionDerivative(float* x, float* y, float* z, float* jacobian, int nParticles, int index);
	__device__ void directSolve(ConstraintArgs args);
	__host__ void directSolve_cpu(float* x, float* y, float* z, float* invmass);
private:
	bool apply_friction;
	float d;
	float muS = 0.001f;
	float muD = 0.0005f;
};