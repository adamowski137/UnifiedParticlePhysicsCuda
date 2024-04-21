#pragma once
#include <memory>
#include <map>
#include <string>

#include "../../GUI/Renderer/Renderer.hpp"
#include "../../GUI/Camera/Camera.hpp"
#include "../../GUI/Input/KeyInput.h"
#include "../../PhysicsEngine/Particle/Particle.cuh"

class Scene
{
protected:
	ParticleType particles;
	std::unique_ptr<Renderer> renderer;
	RenderInfo sceneSphere;
	Camera camera;
	KeyInput input;
	float cameraRadius, cameraAngle;
public:
	Scene(std::shared_ptr<Shader>& shader, int n, void(*setDataFunction)(int, float*, float*, float*, float*, float*, float*, int*), int mode = 0);
	virtual ~Scene();
	virtual void update(float dt);
	void handleKeys();
	virtual void draw();
};