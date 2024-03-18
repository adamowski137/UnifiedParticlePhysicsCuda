#include "TestScene.hpp"
#include "../../ResourceManager/ResourceManager.hpp"
#include "TestScene_data.cuh"

TestScene::TestScene() : Scene(ResourceManager::Instance.Shaders["instancedphong"], 2, initData_TestScene,
	ANY_CONSTRAINTS_ON | GRID_CHECKING_ON)
{

	std::vector<float> offsets;
	offsets.resize(2 * 3, 0.0f);

	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setConstraints({ }, 2.f);
	particles.setExternalForces(0.f, 0.f, 0.f);
}

TestScene::~TestScene()
{
}

void TestScene::update(float dt)
{
	particles.calculateNewPositions(dt);

	glm::vec3 cameraPos = { 0.f, 0.f, -10.f };
	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix(cameraPos));
	renderer->setCameraPosition(cameraPos);
	renderer->setLightSourcePosition(cameraPos);
	camera.updateVectors({ 0.f, 0.f, 1.f });
}

void TestScene::draw()
{
	particles.renderData(sceneSphere.instancingVBO);
	renderer->drawInstanced(sceneSphere, particles.particleCount());
}
