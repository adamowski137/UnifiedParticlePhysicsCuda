#pragma once
#include <cuda_runtime.h>

enum class ConstraintLimitType { EQ, GEQ, LEQ };

class Constraint
{
public:
	__host__ __device__ void init(int n, float compliance, ConstraintLimitType type);
	int n;
	float compliance;
	float cMin;
	float cMax;
};