#include "Scene_NaiveDraw.cuh"
#include "../../ResourceManager/ResourceManager.hpp"

Scene_NaiveDraw::Scene_NaiveDraw() : Scene(ResourceManager::Instance.Shaders["phong"], 1)
{

}

Scene_NaiveDraw::~Scene_NaiveDraw()
{
}

void Scene_NaiveDraw::update(float dt)
{
	glm::vec3 cameraPos = { 0.f, 0.f, -20.f };
	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix());
	renderer->setCameraPosition(cameraPos);
	renderer->setLightSourcePosition(cameraPos);
	camera.updateVectors({ 0.f, 0.f, 1.f });
}

void Scene_NaiveDraw::draw()
{
	for (int i = -30; i < 30; i += 3)
	{
		for (int j = -30; j < 30; j += 3)
		{
			for (int k = 0; k < 30; k += 3)
			{
				renderer->draw(*ResourceManager::Instance.drawData["sphere"], { i, j, k }, { 1, 1, 0 });
			}
		}
	}
}

void Scene_NaiveDraw::initData(int nParticles, float* dev_x, float* dev_y, float* dev_z, 
	float* dev_vx, float* dev_vy, float* dev_vz, 
	int* dev_phase, float* dev_invmass)
{
}
