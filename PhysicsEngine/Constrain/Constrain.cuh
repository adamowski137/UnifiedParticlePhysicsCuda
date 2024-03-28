#pragma once
#include <cuda_runtime.h>

enum class ConstraintLimitType { EQ, GEQ, LEQ };

class Constrain
{
public:
	__host__ __device__ void init(int n, float k, ConstraintLimitType type);
	int n;
	float k;
	float cMin;
	float cMax;
};