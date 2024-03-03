#pragma once

#include "../Scene.hpp"

class TestScene : public Scene
{
public:
	TestScene();
	virtual ~TestScene();
	virtual void update(float dt);
	virtual void draw();
};