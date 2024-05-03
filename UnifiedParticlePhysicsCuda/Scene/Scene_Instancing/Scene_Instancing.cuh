#pragma once

#include "../Scene.hpp"

class Scene_Instancing : public Scene
{
public:
	Scene_Instancing();
	virtual ~Scene_Instancing();
	virtual void update(float dt);
	virtual void draw();
protected:
	virtual void initData(int nParticles,
		float* dev_x, float* dev_y, float* dev_z,
		float* dev_vx, float* dev_vy, float* dev_vz, int* dev_phase, float* dev_invmass) override;
};