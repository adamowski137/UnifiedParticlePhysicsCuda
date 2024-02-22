#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void jaccobiKern(int n, float* A, float* b, float* x, float* outX);
__global__ void gausSeidlKern(int n, float* A, float* b, float* x, float* outX);
__global__ void findGraphColoringKern(int n, float* A, float* colors);

void jaccobi(int n, float* A, float* b, float* x);
void gaussSeidl(int n, float* A, float* b, float* x);
void findGraphColoringKern(int n, float* A, float* colors);
