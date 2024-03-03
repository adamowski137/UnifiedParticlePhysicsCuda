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
	Scene(std::shared_ptr<Shader>& shader, int n);
	virtual ~Scene();
	virtual void update(float dt);
	virtual void draw();
};