#pragma once
#include <cuda_runtime.h>
class Constrain
{
public:
	Constrain(int n, float k, float cMin, float cMax, int* indexes);
	~Constrain();
	int* dev_indexes;
	int n;
	float k;
	float cMin;
	float cMax;
};