#pragma once

#include "../Scene.hpp"


class Cloth_Scene : public Scene
{
public:
	Cloth_Scene();
	virtual ~Cloth_Scene();
	virtual void update(float dt);
	virtual void draw();
	virtual void reset();
protected:
	void initData(int nParticles,
		float* dev_x, float* dev_y, float* dev_z,
		float* dev_vx, float* dev_vy, float* dev_vz, int* dev_phase, float* dev_invmass) override;
};
