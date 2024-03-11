﻿// UnifiedParticlePhysicsCuda.cpp : Defines the entry point for the application.
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
	int n = 6;

	// init glfw, glad
	Window::Instance.initInstance(ResourceManager::Instance.config.width, ResourceManager::Instance.config.height);

	const std::string shaderResPath = "../../../../GUI/res/shaders";

	ResourceManager::Instance.loadAllShaders(shaderResPath);
	ResourceManager::Instance.loadScenes(n);
	
	// semi fixed time step
	// https://gafferongames.com/post/fix_your_timestep/
	float dt = (1.f / 60.f) / 2.f;
	float frameTime = 0.001f;
	while (!Window::Instance.isClosed())
	{
		auto start = std::chrono::high_resolution_clock::now();


		while (dt < frameTime)
		{
			ResourceManager::Instance.getActiveScene()->update(dt);
			frameTime -= dt;
		}

		Window::Instance.clear(255, 255, 255, 1);
		ResourceManager::Instance.getActiveScene()->draw();
		Window::Instance.finishRendering(ResourceManager::Instance.options);

		auto end = std::chrono::high_resolution_clock::now();
		long long ticks = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		frameTime += (float)ticks / 1000000.0f;
	}
	return 0;
}
