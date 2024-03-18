#include "ParticleData.cuh"
#include <device_launch_parameters.h>

__global__ void initializeRandomKern(int amount, curandState* state)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= amount) return;
	curand_init(1234, index, 0, &state[index]);
}

__global__ void fillRandomKern(int amount, float* dst, curandState* state, float min, float max)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= amount) return;
	dst[index] = (max - min) * curand_uniform(&state[index]) + min;
}
