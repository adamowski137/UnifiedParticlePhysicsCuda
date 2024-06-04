#pragma once

#include "../Scene.hpp"
#include "../../PhysicsEngine/Cloth/Cloth.cuh"
#include "../../PhysicsEngine/RigidBodies/RigidBody.cuh"
#include "../../GUI/Renderer/ClothRenderer.hpp"


class Scene_Covering : public Scene
{
public:
	Scene_Covering();
	virtual ~Scene_Covering();
	virtual void update(float dt);
	virtual void draw();
	virtual void reset();
protected:
	void initData(int nParticles,
		float* dev_x, float* dev_y, float* dev_z,
		float* dev_vx, float* dev_vy, float* dev_vz, int* dev_phase, float* dev_invmass) override;
private:
	ClothRenderer clothRenderer;
	RigidBody rigidBody;
	Cloth cloth;
};
