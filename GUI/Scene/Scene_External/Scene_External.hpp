#pragma once

#include "../Scene.hpp"

class Scene_External : public Scene
{
public:
	Scene_External(int amountOfPoints);
	virtual ~Scene_External();
	virtual void update();
	virtual void draw();
	virtual unsigned int getVBO();
};