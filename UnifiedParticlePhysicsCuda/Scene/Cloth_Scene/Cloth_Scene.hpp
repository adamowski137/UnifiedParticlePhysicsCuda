#pragma once

#include "../Scene.hpp"

class Cloth_Scene : public Scene
{
public:
	Cloth_Scene();
	virtual ~Cloth_Scene();
	virtual void update(float dt);
	virtual void draw();
};