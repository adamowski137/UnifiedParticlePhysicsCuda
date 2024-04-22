#include "Scene.hpp"

#include <GLFW/glfw3.h>
#include <cmath>

#include "../ResourceManager/ResourceManager.hpp"
#include "../../GUI/Renderer/MeshGenerator.hpp"
#include <fstream>


Scene::Scene(std::shared_ptr<Shader>& shader, int n, void(*setDataFunction)(int, float*, float*, float*, float*, float*, float*, int*), int mode) :
	camera(ResourceManager::Instance.config.width, ResourceManager::Instance.config.height, { 0, 0, -10 }),
	particles(n, mode, setDataFunction),
	input({ GLFW_KEY_W, GLFW_KEY_S, GLFW_KEY_A, GLFW_KEY_D }, {})
{
	renderer = std::make_unique<Renderer>(shader);
	sceneSphere = getSphereData(10, 10);
	cameraRadius = 10.f;
	cameraAngleHorizontal = cameraAngleVertical = 0.f;
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
	float dA = 0.04, dR = 0.5, verticalLimit = 89.f / 180.f * 3.1415f;

	if (input.getKeyDown(GLFW_KEY_A))
		cameraAngleHorizontal += dA;
	if (input.getKeyDown(GLFW_KEY_D))
		cameraAngleHorizontal -= dA;

	if (input.getKeyDown(GLFW_KEY_W))
	{
		cameraAngleVertical += dA;
		if (cameraAngleVertical > verticalLimit)
			cameraAngleVertical = verticalLimit;
	}
	if (input.getKeyDown(GLFW_KEY_S))
	{
		cameraAngleVertical -= dA;
		if (cameraAngleVertical < -verticalLimit)
			cameraAngleVertical = -verticalLimit;
	}


	int scroll = input.getScrollOffset();
	if (scroll)
	{
		cameraRadius += scroll * dR;
	}

	camera.setPosition(glm::vec3(cameraRadius * sin(cameraAngleHorizontal) * cos(cameraAngleVertical), 
		cameraRadius * sin(cameraAngleVertical),
		-cameraRadius * cos(cameraAngleHorizontal) * cos(cameraAngleVertical)));

	camera.updateVectors(-camera.getPosition());

}

void Scene::draw()
{

}