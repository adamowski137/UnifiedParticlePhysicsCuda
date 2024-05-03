#pragma once

#include "../Scene.hpp"

class TestScene : public Scene
{
public:
	TestScene(int n);
	virtual ~TestScene();
	virtual void update(float dt);
	virtual void draw();
	void initData(int nParticles,
		float* dev_x, float* dev_y, float* dev_z,
		float* dev_vx, float* dev_vy, float* dev_vz, 
		int* dev_phase, float* dev_invmass) override;
};