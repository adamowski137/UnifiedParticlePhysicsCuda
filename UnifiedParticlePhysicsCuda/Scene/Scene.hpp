#pragma once
#include <memory>
#include <map>
#include <string>

#include "../../GUI/Renderer/Renderer.hpp"
#include "../../GUI/Camera/Camera.hpp"
#include "../../PhysicsEngine/Particle/Particle.cuh"


class Scene
{
protected:
	ParticleType particles;
	std::unique_ptr<Renderer> renderer;
	RenderInfo sceneSphere;
	Camera camera;
public:
	Scene(std::shared_ptr<Shader>& shader, int n, void(*setDataFunction)(int, float*, float*, float*, float*, float*, float*), int mode = 0);
	virtual ~Scene();
	virtual void update(float dt);
	virtual void draw();
};