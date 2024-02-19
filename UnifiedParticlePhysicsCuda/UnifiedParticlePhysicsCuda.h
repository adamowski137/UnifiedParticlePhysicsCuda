// UnifiedParticlePhysicsCuda.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>
#include <cuda_runtime.h>

__global__ void addKernel(int* c, const int* a, const int* b);
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

// TODO: Reference additional headers your program requires here.
