#pragma once

#include "../Scene.hpp"

class Scene_External : public Scene
{
public:
	Scene_External(int amountOfPoints);
	virtual ~Scene_External();
	virtual void update(float dt);
	virtual void draw();
};