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
	int n = 4000;
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

	ParticleType particles{ n };


	const std::string shaderResPath = "../../../../GUI/res/shaders";
	const int spherePrecision = 20;

	ResourceManager::Instance.loadAllShaders(shaderResPath);
	ResourceManager::Instance.loadSphereData(spherePrecision, spherePrecision);
	ResourceManager::Instance.loadScenes(n);
	
	
	particles.mapCudaVBO(ResourceManager::Instance.getActiveScene()->getVBO());
	float dt = 0.001f;
	while (!Window::Instance.isClosed())
	{
		auto start = std::chrono::high_resolution_clock::now();

		particles.calculateNewPositions(dt);
		ResourceManager::Instance.getActiveScene()->update();
		particles.renderData(ResourceManager::Instance.getActiveScene()->getVBO());

		Window::Instance.clear(255, 255, 255, 1);
		ResourceManager::Instance.getActiveScene()->draw();
		Window::Instance.finishRendering(ResourceManager::Instance.options);

		auto end = std::chrono::high_resolution_clock::now();
		long long ticks = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		dt = (float)ticks / 1000000.0f;
	}
	return 0;
}
