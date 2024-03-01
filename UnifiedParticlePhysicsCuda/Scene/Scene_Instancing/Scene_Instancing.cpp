#include "Scene_Instancing.hpp"
#include "../../ResourceManager/ResourceManager.hpp"

Scene_Instancing::Scene_Instancing() : Scene(ResourceManager::Instance.Shaders["instancedphong"])
{
	std::vector<float> offsets;
	for (int i = -30; i < 30; i += 3)
	{
		for (int j = -30; j < 30; j += 3)
		{
			for (int k = 0; k < 30; k += 3)
			{
				offsets.push_back(i);
				offsets.push_back(j);
				offsets.push_back(k);
			}
		}
	}

	ResourceManager::Instance.drawData["sphere"]->addInstancing(offsets);
}

Scene_Instancing::~Scene_Instancing()
{
}

void Scene_Instancing::update()
{
	glm::vec3 cameraPos = { 0.f, 0.f, -20.f };
	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix(cameraPos));
	renderer->setCameraPosition(cameraPos);
	renderer->setLightSourcePosition(cameraPos);
	camera.updateVectors({ 0.f, 0.f, 1.f });
}

void Scene_Instancing::draw()
{
	renderer->drawInstanced(*ResourceManager::Instance.drawData["sphere"], 4000);
}
