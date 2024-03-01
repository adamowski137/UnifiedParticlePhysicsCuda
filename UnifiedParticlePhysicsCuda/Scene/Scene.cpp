#include "Scene.hpp"

#include <GLFW/glfw3.h>
#include <cmath>

#include "../ResourceManager/ResourceManager.hpp"
#include <fstream>


Scene::Scene(std::shared_ptr<Shader>& shader) : camera(ResourceManager::Instance.config.width, ResourceManager::Instance.config.height)
{
	renderer = std::make_unique<Renderer>(shader);
}

Scene::~Scene()
{
}

void Scene::update()
{

}

void Scene::draw()
{

}

unsigned int Scene::getVBO()
{
	return 0;
}