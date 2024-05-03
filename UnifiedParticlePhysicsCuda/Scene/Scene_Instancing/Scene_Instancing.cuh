#pragma once

#include "../Scene.hpp"

class Scene_Instancing : public Scene
{
public:
	Scene_Instancing();
	virtual ~Scene_Instancing();
	virtual void update(float dt);
	virtual void draw();
};