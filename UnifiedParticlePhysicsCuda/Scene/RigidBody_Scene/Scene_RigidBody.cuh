#pragma once

#include "../Scene.hpp"

class Scene_RigidBody : public Scene
{
public:
	Scene_RigidBody();
	virtual ~Scene_RigidBody();
	virtual void update(float dt);
	virtual void draw();
	virtual void reset();
private:
	void initData(int nParticles,
		float* dev_x, float* dev_y, float* dev_z,
		float* dev_vx, float* dev_vy, float* dev_vz, int* dev_phase, float* dev_invmass) override;
};