#include "Constrain.cuh"
#include <cuda_runtime.h>
#include "../GpuErrorHandling.hpp"

Constrain::Constrain(int n, float k, float cMin, float cMax, int* indexes) : n{n}, k{k}, cMin{cMin}, cMax{cMax}
{
	gpuErrchk(cudaMalloc((void**)&dev_indexes, n * sizeof(int)));
	gpuErrchk(cudaMemcpy(dev_indexes, indexes, n * sizeof(int), cudaMemcpyHostToDevice));
}

Constrain::~Constrain()
{
	gpuErrchk(cudaFree(dev_indexes));
}
