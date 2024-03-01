#pragma once
#include <memory>
#include <map>
#include <string>

#include "../../GUI/Renderer/Renderer.hpp"
#include "../../GUI/Camera/Camera.hpp"


class Scene
{
protected:
	std::unique_ptr<Renderer> renderer;
	Camera camera;
public:
	Scene(std::shared_ptr<Shader>& shader);
	virtual ~Scene();
	virtual void update();
	virtual void draw();
	virtual unsigned int getVBO();
};