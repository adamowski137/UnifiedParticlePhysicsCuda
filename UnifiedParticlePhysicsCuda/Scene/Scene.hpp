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
	float cameraRadius, cameraAngleHorizontal, cameraAngleVertical;
public:
	Scene(std::shared_ptr<Shader>& shader, int n, int mode = 0);
	virtual ~Scene();
	virtual void update(float dt);
	void handleKeys();
	virtual void draw();
	virtual void reset();
protected:
	void applySceneSetup();
	virtual void initData(int nParticles,
		float* dev_x, float* dev_y, float* dev_z,
		float* dev_vx, float* dev_vy, float* dev_vz,
		int* phase, float* invmass) = 0;
};