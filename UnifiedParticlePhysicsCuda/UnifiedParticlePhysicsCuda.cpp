// UnifiedParticlePhysicsCuda.cpp : Defines the entry point for the application.
//

#include "UnifiedParticlePhysicsCuda.h"
#include "../GUI/Scene/TestScene/TestScene.hpp"
#include "../GUI/Window/Window.hpp"
#include "../GUI/ResourceManager/ResourceManager.hpp"
#include <device_launch_parameters.h>
#include <chrono>
#include "Math/LinearSolver.cuh"
#include "Particle/Particle.cuh"
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

	ParticleType particles{ n };

	Window::getInstance();

	const std::string shaderResPath = "../../../../GUI/res/shaders";
	const int spherePrecision = 20;

	ResourceManager::get().loadAllShaders(shaderResPath);
	ResourceManager::get().loadSphereData(spherePrecision, spherePrecision);
	ResourceManager::get().loadScenes(n);
	
	std::shared_ptr<Scene> currScene = std::shared_ptr<Scene>((*ResourceManager::get().scenes.begin()).second);
	
	particles.mapCudaVBO(currScene->getVBO());
	float dt = 0.001f;
	while (!Window::getInstance().isClosed())
	{
		auto start = std::chrono::high_resolution_clock::now();

		particles.calculateNewPositions(dt);
		currScene->update();
		particles.renderData(currScene->getVBO());

		Window::getInstance().clear(255, 255, 255, 1);
		currScene->draw();
		Window::getInstance().finishRendering(currScene);

		auto end = std::chrono::high_resolution_clock::now();
		long long ticks = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		dt = (float)ticks / 1000000.0f;
	}
	return 0;
}
