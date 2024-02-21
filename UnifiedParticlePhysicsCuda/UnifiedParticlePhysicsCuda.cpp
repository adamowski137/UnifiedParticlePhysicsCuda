// UnifiedParticlePhysicsCuda.cpp : Defines the entry point for the application.
//

#include "UnifiedParticlePhysicsCuda.h"
#include "./../Math/LinearSolver.cuh"
#include <device_launch_parameters.h>
#include <chrono>

using namespace std;

int main()
{
	int n = 10000;
	std::cout << "Matrix size: " << n << std::endl;
	float* A = new float[n * n];
	float* b = new float[n];
	float* x = new float[n];

	for (int i = 0; i < n * n; i++)
	{
		A[i] = rand() % 100;
	}
	for (int i = 0; i < n; i++)
	{
		b[i] = rand() % 100;
		x[i] = 0;
	}
	
	auto start = std::chrono::high_resolution_clock::now();
	jaccobi(n, A, b, x);
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "duration whole: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	return 0;
}
