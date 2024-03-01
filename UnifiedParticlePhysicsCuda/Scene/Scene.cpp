#include "Scene.hpp"

#include <GLFW/glfw3.h>
#include <cmath>

#include "../ResourceManager/ResourceManager.hpp"
#include "../../GUI/Renderer/MeshGenerator.hpp"
#include <fstream>


Scene::Scene(std::shared_ptr<Shader>& shader, int n) : 
	camera(ResourceManager::Instance.config.width, ResourceManager::Instance.config.height),
	particles(n)
{
	renderer = std::make_unique<Renderer>(shader);
	sceneSphere = getSphereData(10, 10);
}

Scene::~Scene()
{
	sceneSphere.dispose();
}

void Scene::update(float dt)
{

}

void Scene::draw()
{

}