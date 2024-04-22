#include "TestScene.hpp"
#include "../../ResourceManager/ResourceManager.hpp"
#include "TestScene_data.cuh"

TestScene::TestScene(int n) : Scene(ResourceManager::Instance.Shaders["instancedphong"], n, initData_TestScene,
	ANY_CONSTRAINTS_ON | GRID_CHECKING_ON | SURFACE_CHECKING_ON)
{

	std::vector<float> offsets;
	offsets.resize(n * 3, 0.0f);

	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setConstraints({ }, 2.f);
	particles.setExternalForces(0.f, -9.81f, 0.f);
	particles.setSurfaces({ Surface().init(0, 1, 0, 10), Surface().init(1, 0, 0, 10), Surface().init(-1, 0, 0, 10), Surface().init(0, 0, 1, 10), Surface().init(0, 0, -1, 10)});

	camera.setPosition(glm::vec3(0, 0, -10));
}

TestScene::~TestScene()
{
}

void TestScene::update(float dt)
{
	particles.calculateNewPositions(dt);
	this->handleKeys();
	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix());
	renderer->setLightSourcePosition({ 0.f, 0.f, -10.f });
}

void TestScene::draw()
{
	particles.renderData(sceneSphere.instancingVBO);
	renderer->drawInstanced(sceneSphere, particles.particleCount());
}
