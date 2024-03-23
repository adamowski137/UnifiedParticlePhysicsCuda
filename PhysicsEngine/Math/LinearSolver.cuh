#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


void jaccobi(int n, float* A, float* b, float* x, float* new_x, float* c_min, float* c_max, int iterations);
