#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void jaccobiKern(int n, float* A, float* b, float* x, float* outX);

void jaccobi(int n, float* A, float* b, float* x);
