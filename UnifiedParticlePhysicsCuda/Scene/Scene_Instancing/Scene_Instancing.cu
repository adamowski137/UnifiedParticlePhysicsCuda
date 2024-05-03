#include "Scene_Instancing.cuh"
#include "../../ResourceManager/ResourceManager.hpp"

Scene_Instancing::Scene_Instancing() : Scene(ResourceManager::Instance.Shaders["instancedphong"], 1)
{
	std::vector<float> offsets;
	for (int i = -60; i < 60; i += 3)
	{
		for (int j = -30; j < 30; j += 3)
		{
			for (int k = 0; k < 60; k += 3)
			{
				offsets.push_back(i);
				offsets.push_back(j);
				offsets.push_back(k);
			}
		}
	}

	sceneSphere.addInstancing(offsets);

	renderer->setSphereColor({ 1.f, 1.f, 0.f });
}

Scene_Instancing::~Scene_Instancing()
{
}

void Scene_Instancing::update(float dt)
{
	glm::vec3 cameraPos = { 0.f, 0.f, -20.f };
	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix());
	renderer->setCameraPosition(cameraPos);
	renderer->setLightSourcePosition(cameraPos);
	camera.updateVectors({ 0.f, 0.f, 1.f });
}

void Scene_Instancing::draw()
{
	renderer->drawInstanced(sceneSphere, 32000);
}

void Scene_Instancing::initData(int nParticles, float* dev_x, float* dev_y, float* dev_z, float* dev_vx, float* dev_vy, float* dev_vz, int* dev_phase, float* dev_invmass)
{
}
