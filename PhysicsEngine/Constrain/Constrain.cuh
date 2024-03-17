#pragma once
#include <cuda_runtime.h>

__host__ __device__ enum class ConstraintLimitType { EQ, GEQ, LEQ };

class Constrain
{
public:
	__host__ __device__ Constrain(int n, float k, ConstraintLimitType type);
	__host__ __device__ ~Constrain();
	int n;
	float k;
	float cMin;
	float cMax;
};