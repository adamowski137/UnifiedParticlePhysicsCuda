#include "Scene.hpp"

#include <GLFW/glfw3.h>
#include <cmath>

#include "../Error/ErrorHandling.hpp"
#include "../Renderer/MeshGenerator.hpp"
#include "../ResourceManager/ResourceManager.hpp"
#include <fstream>


Scene::Scene(std::shared_ptr<Shader>& shader) : camera(ResourceManager::get().config.width, ResourceManager::get().config.height)
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