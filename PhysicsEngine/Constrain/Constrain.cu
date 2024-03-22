#include "Constrain.cuh"
#include <cuda_runtime.h>
#include "../GpuErrorHandling.hpp"
#include <iostream>
#include <climits>

__host__ __device__ Constrain::Constrain(int n, float k, ConstraintLimitType type) : n{n}, k{k}
{
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

	//gpuErrchk(cudaMalloc((void**)&dev_indexes, n * sizeof(int)));
	//gpuErrchk(cudaMemcpy(dev_indexes, indexes, n * sizeof(int), cudaMemcpyHostToDevice));
	//int tmp[2];
	//gpuErrchk(cudaMemcpy(tmp, dev_indexes, n * sizeof(int), cudaMemcpyDeviceToHost));
}

__host__ __device__ Constrain::Constrain() : n{ 0 }, k{ 0 }, cMin{ 0 }, cmax{ 0 }
{
}
