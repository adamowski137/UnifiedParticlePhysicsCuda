#pragma once

#include "../Scene.hpp"

class Scene_External : public Scene
{
public:
	Scene_External(int amountOfPoints);
	virtual ~Scene_External();
	virtual void update(float dt);
	virtual void draw();
	static void initData_SceneExternal(int nParticles,
		float* dev_x, float* dev_y, float* dev_z,
		float* dev_vx, float* dev_vy, float* dev_vz, int* mode);
};