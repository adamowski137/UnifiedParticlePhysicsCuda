#include "Constraint.cuh"
#include <cuda_runtime.h>
#include "../GpuErrorHandling.hpp"
#include <iostream>
#include <climits>

__host__ __device__ void Constraint::init(int n, float k, ConstraintLimitType type)
{
	this->n = n;
	this->k = k;
	const float max = 1e15;
	const float min = -1e15;
	switch (type)
	{
	case ConstraintLimitType::EQ:
		cMax = max;
		cMin = min;
		break;
	case ConstraintLimitType::GEQ:
		cMax = max;
		cMin = 0;
		break;
	case ConstraintLimitType::LEQ:
		cMax = 0;
		cMin = min;
		break;
	default:
		break;
	}
}
