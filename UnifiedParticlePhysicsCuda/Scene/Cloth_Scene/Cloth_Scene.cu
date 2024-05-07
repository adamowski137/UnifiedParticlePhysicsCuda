#include "Cloth_Scene.cuh"
#include "../../ResourceManager/ResourceManager.hpp"
#include "../../PhysicsEngine/Cloth/Cloth.hpp"

#include "../../PhysicsEngine/Particle/ParticleData.cuh"
#include "../../PhysicsEngine/GpuErrorHandling.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#define CLOTH_W 10
#define CLOTH_H 12
#define NUM_PARTICLES (CLOTH_W * CLOTH_H + 1)

Cloth_Scene::Cloth_Scene() :
	Scene(ResourceManager::Instance.Shaders["instancedphong"], NUM_PARTICLES, ANY_CONSTRAINTS_ON | GRID_CHECKING_ON)
{
	std::vector<float> offsets;
	offsets.resize(NUM_PARTICLES * 3, 0.0f);

	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setExternalForces(0.f, -9.81f, -10.f);
	particles.setRigidBodyConstraint({});

	camera.setPosition(glm::vec3(0, 0, -10));

	applySceneSetup();
	ConstraintStorage<DistanceConstraint>::Instance.setStaticConstraints(Cloth::getConstraints().first, Cloth::getConstraints().second);
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

void Cloth_Scene::reset()
{
	particles.clearConstraints();
	ConstraintStorage<DistanceConstraint>::Instance.setStaticConstraints(Cloth::getConstraints().first, Cloth::getConstraints().second);
}

void Cloth_Scene::initData(int nParticles, float* dev_x, float* dev_y, float* dev_z, float* dev_vx, float* dev_vy, float* dev_vz, int* dev_phase, float* dev_invmass)
{
	curandState* dev_curand;
	int threads = 32;

	int blocks = (nParticles + threads - 1) / threads;
	gpuErrchk(cudaMalloc((void**)&dev_curand, nParticles * sizeof(curandState)));

	initializeRandomKern << < blocks, threads >> > (nParticles, dev_curand);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	float d = 2.1f;
	int W = CLOTH_W;
	int H = CLOTH_H;
	Cloth::initClothSimulation(H, W, d, -d * W / 2.f, 0.f, 0.f, dev_x, dev_y, dev_z);

	std::vector<float> invmass(nParticles, 1.f);
	invmass[0] = 0.f;
	invmass[W - 1] = 0.f;
	gpuErrchk(cudaMemcpy(dev_invmass, invmass.data(), invmass.size() * sizeof(float), cudaMemcpyHostToDevice));


	fillRandomKern << <blocks, threads >> > (nParticles, dev_z, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vx, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vy, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vz, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	// set the colliding particle
	auto xptr = thrust::device_pointer_cast(dev_x);
	auto yptr = thrust::device_pointer_cast(dev_y);
	auto zptr = thrust::device_pointer_cast(dev_z);

	auto vzptr = thrust::device_pointer_cast(dev_vz);

	xptr[NUM_PARTICLES - 1] = 0.f;
	yptr[NUM_PARTICLES - 1] = -5.f;
	zptr[NUM_PARTICLES - 1] = -50.f;
	vzptr[NUM_PARTICLES - 1] = 50.f;

	cudaFree(dev_curand);
}
