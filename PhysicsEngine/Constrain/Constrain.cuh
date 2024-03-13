#pragma once
#include <cuda_runtime.h>

enum class ConstraintLimitType { EQ, GEQ, LEQ };

class Constrain
{
public:
	Constrain(int n, float k, ConstraintLimitType type);
	~Constrain();
	int n;
	float k;
	float cMin;
	float cMax;
};