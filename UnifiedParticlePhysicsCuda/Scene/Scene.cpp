#include "Scene.hpp"

#include <GLFW/glfw3.h>
#include <cmath>

#include "../ResourceManager/ResourceManager.hpp"
#include "../../GUI/Renderer/MeshGenerator.hpp"
#include <fstream>


Scene::Scene(std::shared_ptr<Shader>& shader, int n, void(*setDataFunction)(int, float*, float*, float*, float*, float*, float*, int*), int mode) :
	camera(ResourceManager::Instance.config.width, ResourceManager::Instance.config.height, { 0, 0, -10 }),
	particles(n, mode, setDataFunction),
	input({ GLFW_KEY_A, GLFW_KEY_D }, {})
{
	renderer = std::make_unique<Renderer>(shader);
	sceneSphere = getSphereData(10, 10);
	cameraRadius = 10.f;
	cameraAngle = 0.f;
}

Scene::~Scene()
{
	sceneSphere.dispose();
}

void Scene::update(float dt)
{

}

void Scene::handleKeys()
{
	glm::vec3 currCameraPosition = camera.getPosition();
	float dA = 0.04, dR = 0.5;
	if (input.getKeyDown(GLFW_KEY_A))
		cameraAngle += dA;
	if (input.getKeyDown(GLFW_KEY_D))
		cameraAngle -= dA;


	int scroll = input.getScrollOffset();
	if (scroll)
	{
		cameraRadius += scroll * dR;
	}

	camera.setPosition(glm::vec3(cameraRadius * sin(cameraAngle), currCameraPosition.y, cameraRadius * -cos(cameraAngle)));
	camera.updateVectors(glm::vec3(-sin(cameraAngle), 0, cos(cameraAngle)));

}

void Scene::draw()
{

}