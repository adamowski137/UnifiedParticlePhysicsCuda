#pragma once

#include "../Scene.hpp"

class Scene_NaiveDraw : public Scene
{
public:
	Scene_NaiveDraw();
	virtual ~Scene_NaiveDraw();
	virtual void update();
	virtual void draw();
};