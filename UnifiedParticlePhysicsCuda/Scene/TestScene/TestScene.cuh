#pragma once

#include "../Scene.hpp"

class TestScene : public Scene
{
public:
	TestScene(int n);
	virtual ~TestScene();
	virtual void update(float dt);
	virtual void draw();
	static void initData_TestScene(int nParticles,
		float* dev_x, float* dev_y, float* dev_z,
		float* dev_vx, float* dev_vy, float* dev_vz, int* mode);
};