#include "Cloth_Scene.cuh"
#include "../../ResourceManager/ResourceManager.hpp"

#include "../../PhysicsEngine/Particle/ParticleData.cuh"
#include "../../PhysicsEngine/GpuErrorHandling.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <vector>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#define CLOTH_W 70
#define CLOTH_H 70
#define N_RIGID_BODY 5
#define NUM_PARTICLES (CLOTH_W * CLOTH_H + N_RIGID_BODY * N_RIGID_BODY * N_RIGID_BODY)

Cloth_Scene::Cloth_Scene() :
	Scene(ResourceManager::Instance.Shaders["instancedphong"], NUM_PARTICLES, ANY_CONSTRAINTS_ON | GRID_CHECKING_ON),
	clothRenderer(ResourceManager::Instance.Shaders["cloth"])
{
	std::vector<float> offsets;
	offsets.resize(NUM_PARTICLES * 3, 0.0f);


	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setExternalForces(0.f, -98.1f, -40.f);

	camera.setPosition(glm::vec3(0, 0, -10));

	applySceneSetup();
	ConstraintStorage<DistanceConstraint>::Instance.addStaticConstraints(cloth.getConstraints().first, cloth.getConstraints().second);
	particles.mapCudaVBO(cloth.clothMesh.VBO); 
}

Cloth_Scene::~Cloth_Scene()
{
}

void Cloth_Scene::update(float dt)
{
	if(!isPaused)	particles.calculateNewPositions(dt);
	this->handleKeys();

	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix());
	renderer->setCameraPosition(camera.getPosition());
	renderer->setLightSourcePosition(glm::vec3(0, 0, -10));

	clothRenderer.getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix());
	//clothRenderer.setCameraPosition(camera.getPosition());
	//clothRenderer.setLightSourcePosition(glm::vec3(0, 0, -10));
}

void Cloth_Scene::draw()
{
	particles.sendDataToVBO(sceneSphere.instancingVBO, CLOTH_W * CLOTH_H, NUM_PARTICLES - CLOTH_W * CLOTH_H);
	particles.sendDataToVBO(cloth.clothMesh.VBO, 0, CLOTH_W * CLOTH_H);
	renderer->drawInstanced(sceneSphere, NUM_PARTICLES - CLOTH_W * CLOTH_H);
	clothRenderer.draw(cloth.clothMesh);
}

void Cloth_Scene::reset()
{
	particles.clearConstraints();
	ConstraintStorage<RigidBodyConstraint>::Instance.setCpuConstraints(rigidBody.getConstraints());
	ConstraintStorage<DistanceConstraint>::Instance.addStaticConstraints(cloth.getConstraints().first, cloth.getConstraints().second);
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
	Cloth::initClothSimulation_LRA(cloth, H, W, d, -d * W / 2.f, 0.f, 0.f, dev_x, dev_y, dev_z, dev_phase, ClothOrientation::XY_PLANE, {0, W - 1});

	std::vector<float> invmass(CLOTH_W * CLOTH_H, 1.f);
	invmass[0] = 0.f;
	invmass[W - 1] = 0.f;
	gpuErrchk(cudaMemcpy(dev_invmass, invmass.data(), invmass.size() * sizeof(float), cudaMemcpyHostToDevice));

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vx, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vy, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vz, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	rigidBody.addRigidBodySquare(dev_x, dev_y, dev_z, dev_invmass, CLOTH_W * CLOTH_H, N_RIGID_BODY, 0, -20, -80, dev_phase, 3);

	auto vz_ptr = thrust::device_pointer_cast(dev_vz);

	thrust::fill(vz_ptr + CLOTH_W * CLOTH_H, vz_ptr + nParticles, 100.0f);


	cudaFree(dev_curand);
}
