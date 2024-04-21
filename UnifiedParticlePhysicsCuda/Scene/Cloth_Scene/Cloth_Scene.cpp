#include "Cloth_Scene.hpp"
#include "Cloth_Scene_data.cuh"
#include "../../ResourceManager/ResourceManager.hpp"

#define CLOTH_SIZE 12

Cloth_Scene::Cloth_Scene() :
	Scene(ResourceManager::Instance.Shaders["instancedphong"], CLOTH_SIZE, initData_ClothScene, ANY_CONSTRAINTS_ON)
{
	std::vector<float> offsets;
	offsets.resize(CLOTH_SIZE * 3, 0.0f);

	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setExternalForces(0.f, -0.5f, 0.f);

	camera.setPosition(glm::vec3(0, 0, -10));
}

Cloth_Scene::~Cloth_Scene()
{
}

void Cloth_Scene::update(float dt)
{
	particles.calculateNewPositions(dt);
	this->handleKeys();

	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix());
	renderer->setCameraPosition(camera.getPosition());
	renderer->setLightSourcePosition(glm::vec3(0, 0, -10));
}

void Cloth_Scene::draw()
{
	particles.renderData(sceneSphere.instancingVBO);
	renderer->drawInstanced(sceneSphere, particles.particleCount());
}
