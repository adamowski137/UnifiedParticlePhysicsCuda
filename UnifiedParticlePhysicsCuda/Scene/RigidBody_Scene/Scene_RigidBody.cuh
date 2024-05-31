#pragma once

#include "../Scene.hpp"
#include "../../PhysicsEngine/RigidBodies/RigidBody.cuh"

class Scene_RigidBody : public Scene
{
public:
	Scene_RigidBody();
	virtual ~Scene_RigidBody();
	virtual void update(float dt);
	virtual void draw();
	virtual void reset();
private:
	RigidBody rigidBody;
	void initData(int nParticles,
		float* dev_x, float* dev_y, float* dev_z,
		float* dev_vx, float* dev_vy, float* dev_vz,
		int* dev_SDF_mode, float* dev_SDF_value, float* dev_SDF_normal_x, float* dev_SDF_normal_y, float* dev_SDF_normal_z,
		int* phase, float* invmass) override;
};