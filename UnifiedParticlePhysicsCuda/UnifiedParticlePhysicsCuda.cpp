// UnifiedParticlePhysicsCuda.cpp : Defines the entry point for the application.
//

#include "UnifiedParticlePhysicsCuda.h"
#include "Scene/TestScene/TestScene.hpp"
#include "../GUI/Window/Window.hpp"
#include "ResourceManager/ResourceManager.hpp"
#include <device_launch_parameters.h>
#include <chrono>
#include "../PhysicsEngine/Math/LinearSolver.cuh"
#include "../PhysicsEngine/Particle/Particle.cuh"
using namespace std;

int main()
{
	int n = 10;
	//std::cout << "Matrix size: " << n << std::endl;
	//float* A = new float[n * n];
	//float* b = new float[n];
	//float* x = new float[n];

	//for (int i = 0; i < n * n; i++)
	//{
	//	A[i] = rand() % 100;
	//}
	//for (int i = 0; i < n; i++)
	//{
	//	b[i] = rand() % 100;
	//	x[i] = 0;
	//}
	//
	//auto start = std::chrono::high_resolution_clock::now();
	//jaccobi(n, A, b, x);
	//auto end = std::chrono::high_resolution_clock::now();
	//std::cout << "duration whole: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	// init glfw, glad
	Window::Instance.initInstance(ResourceManager::Instance.config.width, ResourceManager::Instance.config.height);

	const std::string shaderResPath = "../../../../GUI/res/shaders";

	ResourceManager::Instance.loadAllShaders(shaderResPath);
	ResourceManager::Instance.loadScenes(n);
	
	// semi fixed time step
	// https://gafferongames.com/post/fix_your_timestep/
	float t = 0.f;
	float dt = (1.f / 60.f) / 4.f;
	float accumulator = 0.f;
	auto current_time = std::chrono::high_resolution_clock::now();
	while (!Window::Instance.isClosed())
	{
		auto new_time = std::chrono::high_resolution_clock::now();
		float frameTime = std::chrono::duration_cast<std::chrono::microseconds>(new_time - current_time).count() / 1000000.f;
		current_time = new_time;

		accumulator += frameTime;

		while (accumulator > dt)
		{
			ResourceManager::Instance.getActiveScene()->update(dt);
			accumulator -= dt;
			t += dt;
		}

		Window::Instance.clear(255, 255, 255, 1);
		ResourceManager::Instance.getActiveScene()->draw();
		Window::Instance.finishRendering(ResourceManager::Instance.options);
	}
	return 0;
}
