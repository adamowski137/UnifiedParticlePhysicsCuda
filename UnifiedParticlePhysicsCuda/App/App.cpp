#include "App.hpp"

#include <GLFW/glfw3.h>
#include <cmath>

#include "../Error/ErrorHandling.hpp"
#include "../Renderer/MeshGenerator.hpp"


App::App(int width, int height) : camera(width, height)
{
	Call(glEnable(GL_TEXTURE_2D));
	Call(glEnable(GL_DEPTH_TEST));
	Call(glEnable(GL_CULL_FACE));
	Call(glCullFace(GL_BACK));

	shaders.insert(std::make_pair("colorShader", std::make_shared<Shader>()));
	shaders["colorShader"]->createFromFile("Color");
	renderEntities.insert(std::make_pair("sphere", std::make_unique<RenderInfo>(getSphereData())));

	renderer = std::make_unique<Renderer>(shaders["colorShader"]);
}

App::~App()
{
}

void App::clear(float r, float g, float b, float a)
{
	glClearColor(r / 255.0f, g / 255.0f, b / 255.0f, a);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void App::update()
{
	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix({ 0, 0, -20 }));

}

void App::draw()
{
	renderer->draw(*renderEntities["sphere"].get());
}