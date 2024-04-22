#include "Cloth_Scene.hpp"
#include "Cloth_Scene_data.cuh"
#include "../../ResourceManager/ResourceManager.hpp"
#include "../../PhysicsEngine/Cloth/Cloth.hpp"

#define CLOTH_SIZE 120

Cloth_Scene::Cloth_Scene() :
	Scene(ResourceManager::Instance.Shaders["instancedphong"], CLOTH_SIZE, initData_ClothScene, ANY_CONSTRAINTS_ON)
{
	std::vector<float> offsets;
	offsets.resize(CLOTH_SIZE * 3, 0.0f);

	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setExternalForces(0.f, -9.81f, -20.f);

	camera.setPosition(glm::vec3(0, 0, -10));
}

Cloth_Scene::~Cloth_Scene()
{
}

void Cloth_Scene::update(float dt)
{
	ConstraintStorage<DistanceConstraint>::Instance.setDynamicConstraints(Cloth::getConstraints().first, Cloth::getConstraints().second);
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