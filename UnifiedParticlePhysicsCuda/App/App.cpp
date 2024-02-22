#include "App.hpp"

#include <GLFW/glfw3.h>
#include <cmath>

#include "../Error/ErrorHandling.hpp"
#include "../Renderer/MeshGenerator.hpp"
#include <fstream>

App::App(int width, int height) : camera(width, height)
{
	Call(glEnable(GL_DEPTH_TEST));
	Call(glEnable(GL_CULL_FACE));
	// TODO: naprawic
	Call(glFrontFace(GL_CW));
	Call(glCullFace(GL_BACK));

	shaders.insert(std::make_pair("colorShader", std::make_shared<Shader>()));
	shaders["colorShader"]->createFromFile("./../../../../res/shaders/colorShader/color");
	renderEntities.insert(std::make_pair("sphere", std::make_unique<RenderInfo>(getSphereData())));

	renderer = std::make_unique<Renderer>(shaders["colorShader"]);
}

App::~App()
{
}

void App::update()
{
	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix({ 0, 0, -2.f }));
	camera.updateVectors({ 0.f, 0.f, 1.f });
}

void App::draw()
{
	renderer->draw(*renderEntities["sphere"].get());
}