#pragma once
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void initializeRandomKern(int amount, curandState* state);
__global__ void fillRandomKern(int amount, float* dst, curandState* state, float min, float max);
