#include "Scene_External.hpp"
#include "../../ResourceManager/ResourceManager.hpp"

Scene_External::Scene_External(int amountOfPoints) : Scene(ResourceManager::Instance.Shaders["instancedphong"])
{
	std::vector<float> offsets;
	offsets.resize(amountOfPoints * 3, 0.0f);

	ResourceManager::Instance.drawData["sphere"]->addInstancing(offsets);
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
	renderer->drawInstanced(*ResourceManager::Instance.drawData["sphere"], 4000);
}

unsigned int Scene_External::getVBO()
{
	return ResourceManager::Instance.drawData["sphere"].get()->instancingVBO;
}
