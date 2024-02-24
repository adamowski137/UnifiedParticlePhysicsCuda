#include "Scene_External.hpp"
#include "../../ResourceManager/ResourceManager.hpp"

Scene_External::Scene_External() : Scene(ResourceManager::get().Shaders["instancedphong"])
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
	
	ResourceManager::get().drawData["sphere"]->addInstancing(offsets);
}

Scene_External::~Scene_External()
{
}

void Scene_External::update()
{
	glm::vec3 cameraPos = { 0.f, 0.f, -20.f };
	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix(cameraPos));
	renderer->setCameraPosition(cameraPos);
	renderer->setLightSourcePosition(cameraPos);
	camera.updateVectors({ 0.f, 0.f, 1.f });


}

void Scene_External::draw()
{
	renderer->drawInstanced(*ResourceManager::get().drawData["sphere"], 4000);
}

unsigned int Scene_External::getVBO()
{
	return ResourceManager::get().drawData["sphere"].get()->instancingVBO;
}
