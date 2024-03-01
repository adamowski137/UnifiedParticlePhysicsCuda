#pragma once

#include "../Scene.hpp"

class TestScene : public Scene
{
public:
	TestScene();
	virtual ~TestScene();
	virtual void update();
	virtual void draw();
};