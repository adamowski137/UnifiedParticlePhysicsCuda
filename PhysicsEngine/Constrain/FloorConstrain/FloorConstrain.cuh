#pragma once
#include "../Constrain.cuh"

class FloorConstrain : public Constrain
{
public:
	FloorConstrain(float d, int* indexes);
	__host__ __device__ float operator()(float* x, float* y, float* z,
		float* vx, float* vy, float* vz);
	__host__ __device__  float timeDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz);
	__host__ __device__ void positionDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz, int index, float* output);
	__host__ __device__  void timePositionDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz, int index, float* output);
private:
	float d;
};