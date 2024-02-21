#pragma once
#include <memory>
#include <map>
#include <string>

#include "../Renderer/Renderer.hpp"
#include "../Camera/Camera.hpp"


class App
{
	std::map<std::string, std::shared_ptr<Shader>> shaders;
	std::map<std::string, std::unique_ptr<RenderInfo>> renderEntities;
	std::unique_ptr<Renderer> renderer;
	Camera camera;

public:
	App(int width, int height);
	~App();
	void clear(float r, float g, float b, float a);
	void update();
	void draw();
};