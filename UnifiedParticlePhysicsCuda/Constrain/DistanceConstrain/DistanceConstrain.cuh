#pragma once
#include "../Constrain.cuh"

class DistanceConstrain : public Constrain
{
public:
	DistanceConstrain(float d, int* indexes);
	__host__ __device__ virtual float operator()(float* x, float* y, float* z,
		float* vx, float* vy, float* vz) override;
	__host__ __device__ virtual float timeDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz) override;
	__host__ __device__ virtual float positionDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz, int index) override;
	__host__ __device__ virtual float timePositionDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz, int index) override;
private:
	float d;
};