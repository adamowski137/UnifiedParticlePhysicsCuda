#include "Cloth_Scene.cuh"
#include "../../ResourceManager/ResourceManager.hpp"
#include "../../PhysicsEngine/Cloth/Cloth.hpp"

#include "../../PhysicsEngine/Particle/ParticleData.cuh"
#include "../../PhysicsEngine/GpuErrorHandling.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#define CLOTH_SIZE 20

Cloth_Scene::Cloth_Scene() :
	Scene(ResourceManager::Instance.Shaders["instancedphong"], CLOTH_SIZE, ANY_CONSTRAINTS_ON)
{
	std::vector<float> offsets;
	offsets.resize(CLOTH_SIZE * 3, 0.0f);

	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setExternalForces(0.f, -9.81f, 20.f);

	camera.setPosition(glm::vec3(0, 0, -10));

	applySceneSetup(initData_ClothScene);
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

void Cloth_Scene::initData_ClothScene(int nParticles, float* dev_x, float* dev_y, float* dev_z, float* dev_vx, float* dev_vy, float* dev_vz, int* mode)
{
	curandState* dev_curand;
	int threads = 32;

	int blocks = (nParticles + threads - 1) / threads;
	gpuErrchk(cudaMalloc((void**)&dev_curand, nParticles * sizeof(curandState)));

	initializeRandomKern << < blocks, threads >> > (nParticles, dev_curand);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	float d = 2;
	int W = 10;
	int H = 2;
	Cloth::initClothSimulation(H, W, d, -d * W / 2.f, 0.f, 0.f, dev_x, dev_y, dev_z);

	std::vector<int> modes(nParticles, 0);
	//for (int i = 0; i < W; i++)
	//	modes[i] = 1;
	modes[0] = 1;
	modes[W - 1] = 1;
	gpuErrchk(cudaMemcpy(mode, modes.data(), modes.size() * sizeof(int), cudaMemcpyHostToDevice));


	fillRandomKern << <blocks, threads >> > (nParticles, dev_z, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vx, dev_curand, -0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vy, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}
