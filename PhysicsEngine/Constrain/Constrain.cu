#include "Constrain.cuh"
#include <cuda_runtime.h>
#include "../GpuErrorHandling.hpp"
#include <iostream>
#include <climits>

Constrain::Constrain(int n, float k, ConstraintLimitType type) : n{n}, k{k}
{
	switch (type)
	{
	case ConstraintLimitType::EQ:
		cMax = FLT_MAX;
		cMin = FLT_MIN;
		break;
	case ConstraintLimitType::GEQ:
		cMax = FLT_MAX;
		cMin = 0;
		break;
	case ConstraintLimitType::LEQ:
		cMax = 0;
		cMin = FLT_MIN;
		break;
	default:
		break;
	}

	//gpuErrchk(cudaMalloc((void**)&dev_indexes, n * sizeof(int)));
	//gpuErrchk(cudaMemcpy(dev_indexes, indexes, n * sizeof(int), cudaMemcpyHostToDevice));
	//int tmp[2];
	//gpuErrchk(cudaMemcpy(tmp, dev_indexes, n * sizeof(int), cudaMemcpyDeviceToHost));
}

Constrain::~Constrain()
{
	// MEMORY LEAK!!! 
	//gpuErrchk(cudaFree(dev_indexes));
}
