#include <chrono>
#include "Scene_External.hpp"
#include "Scene_External_data.cuh"
#include "../../ResourceManager/ResourceManager.hpp"

Scene_External::Scene_External(int amountOfPoints) : Scene(
	ResourceManager::Instance.Shaders["instancedphong"], amountOfPoints, initData_SceneExternal, ANY_CONSTRAINTS_ON | SURFACE_CHECKING_ON | GRID_CHECKING_ON)
{
	std::vector<float> offsets;
	offsets.resize(amountOfPoints * 3, 0.0f);

	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setConstraints({ }, 2.f);
	particles.setExternalForces(0.f, -9.81f, 0.f);
	particles.setSurfaces({ Surface().init(0, 1, 0, 0), Surface().init(1, 0, 0, 20), Surface().init(-1, 0, 0, 20)});
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