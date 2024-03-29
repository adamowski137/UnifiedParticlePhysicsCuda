#pragma once
#include "../Constrain.cuh"

class DistanceConstrain : public Constrain
{
public:
	int p[2];
	__host__ __device__ DistanceConstrain init(float d, int p1, int p2, ConstraintLimitType type);
	__host__ __device__ float operator()(float* x, float* y, float* z,
		float* vx, float* vy, float* vz);
	__host__ __device__ void positionDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz, int index, float* output);
private:
	float d;
};