#pragma once
#include <cuda_runtime.h>
class Constrain
{
public:
	Constrain(int n, float k, float cMin, float cMax, int* indexes);
	~Constrain();
	
	__device__ __host__ virtual float operator()(float* x, float* y, float* z,
		float* vx, float* vy, float* vz) = 0;
	__device__ __host__ virtual float timeDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz) = 0;
	__device__ __host__ virtual float positionDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz, int index) = 0;
	__device__ __host__ virtual float timePositionDerivative(float* x, float* y, float* z,
		float* vx, float* vy, float* vz, int index) = 0;

	int* dev_indexes;
	int n;
	float k;
	float cMin;
	float cMax;
};