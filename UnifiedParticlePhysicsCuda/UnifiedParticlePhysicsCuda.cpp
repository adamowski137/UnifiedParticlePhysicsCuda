// UnifiedParticlePhysicsCuda.cpp : Defines the entry point for the application.
//

#include "UnifiedParticlePhysicsCuda.h"
#include "../GUI/App/App.hpp"
#include "../GUI/Window/Window.hpp"
#include <device_launch_parameters.h>
#include <chrono>
#include "Math/LinearSolver.cuh"

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

	// init glfw, glad
	Window::getInstance();

	App app(1024, 768);
	while (!Window::getInstance().isClosed())
	{
		app.update();

		Window::getInstance().clear(255, 255, 255, 1);
		app.draw();
		Window::getInstance().finishRendering();
	}
	return 0;
}
