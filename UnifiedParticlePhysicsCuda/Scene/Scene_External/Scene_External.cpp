#include "Scene_External.hpp"
#include "../../ResourceManager/ResourceManager.hpp"

Scene_External::Scene_External(int amountOfPoints) : Scene(ResourceManager::Instance.Shaders["instancedphong"], amountOfPoints)
{
	std::vector<float> offsets;
	offsets.resize(amountOfPoints * 3, 0.0f);

	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setConstraints({ {0, 1}, {1, 2}, {0, 2}, {0,3}, {1,3}, {2,3} }, 4.f);
}

Scene_External::~Scene_External()
{
}

void Scene_External::update(float dt)
{
	particles.calculateNewPositions(dt);

	glm::vec3 cameraPos = { 0.f, 0.f, -10.f };
	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix(cameraPos));
	renderer->setCameraPosition(cameraPos);
	renderer->setLightSourcePosition(cameraPos);
	camera.updateVectors({ 0.f, 0.f, 1.f });


}

void Scene_External::draw()
{
	particles.renderData(sceneSphere.instancingVBO);
	renderer->drawInstanced(sceneSphere, particles.particleCount());
}