#include "Scene_Trampoline.cuh"
#include "../../ResourceManager/ResourceManager.hpp"

#include "../../PhysicsEngine/Particle/ParticleData.cuh"
#include "../../PhysicsEngine/GpuErrorHandling.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#define CLOTH_W 1000
#define CLOTH_H 1000
#define N_RIGID_BODY 6
#define NUM_PARTICLES (CLOTH_W * CLOTH_H + N_RIGID_BODY * N_RIGID_BODY * N_RIGID_BODY)

Scene_Trampoline::Scene_Trampoline() :
	Scene(ResourceManager::Instance.Shaders["instancedphong"], NUM_PARTICLES, ANY_CONSTRAINTS_ON | GRID_CHECKING_ON),
	clothRenderer(ResourceManager::Instance.Shaders["cloth"])
{
	std::vector<float> offsets;
	offsets.resize(NUM_PARTICLES * 3, 0.0f);

	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setExternalForces(0.f, -98.1f, 0.f);
	particles.setSurfaces({ });

	camera.setPosition(glm::vec3(0, 0, -10));

	applySceneSetup();
	ConstraintStorage<DistanceConstraint>::Instance.addStaticConstraints(cloth.getConstraints().first, cloth.getConstraints().second);
	ConstraintStorage<RigidBodyConstraint>::Instance.setCpuConstraints(rigidBody.getConstraints());

	particles.mapCudaVBO(cloth.clothMesh.VBO);
}

Scene_Trampoline::~Scene_Trampoline()
{
}

void Scene_Trampoline::update(float dt)
{
	if (isPaused)	particles.calculateNewPositions(dt);
	this->handleKeys();

	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix());
	renderer->setCameraPosition(camera.getPosition());
	renderer->setLightSourcePosition(glm::vec3(0, 0, -10));

	clothRenderer.getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix());
}

void Scene_Trampoline::draw()
{
	//particles.sendDataToVBO(sceneSphere.instancingVBO);
	//renderer->drawInstanced(sceneSphere, particles.particleCount());
	particles.sendDataToVBO(sceneSphere.instancingVBO, CLOTH_W * CLOTH_H, NUM_PARTICLES - CLOTH_W * CLOTH_H);
	particles.sendDataToVBO(cloth.clothMesh.VBO, 0, CLOTH_W * CLOTH_H);
	renderer->drawInstanced(sceneSphere, NUM_PARTICLES - CLOTH_W * CLOTH_H);
	clothRenderer.draw(cloth.clothMesh);
}

void Scene_Trampoline::reset()
{
	particles.clearConstraints();
	ConstraintStorage<RigidBodyConstraint>::Instance.setCpuConstraints(rigidBody.getConstraints());
	ConstraintStorage<DistanceConstraint>::Instance.addStaticConstraints(cloth.getConstraints().first, cloth.getConstraints().second);
}

void Scene_Trampoline::initData(int nParticles, float* dev_x, float* dev_y, float* dev_z, float* dev_vx, float* dev_vy, float* dev_vz, int* dev_phase, float* dev_invmass)
{
	curandState* dev_curand;
	int threads = 32;

	int blocks = (nParticles + threads - 1) / threads;
	gpuErrchk(cudaMalloc((void**)&dev_curand, nParticles * sizeof(curandState)));

	initializeRandomKern << < blocks, threads >> > (nParticles, dev_curand);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	float d = 0.5f;
	int W = CLOTH_W;
	int H = CLOTH_H;
	rigidBody.addRigidBodySquare(dev_x, dev_y, dev_z, dev_invmass, CLOTH_W * CLOTH_H, N_RIGID_BODY, -10, 20, -5, dev_phase, 3);
	Cloth::initClothSimulation_simple(cloth, H, W, d, -d * W / 2.f, 10.f, 15.f, dev_x, dev_y, dev_z, dev_phase, ClothOrientation::XZ_PLANE);

	std::vector<float> invmass(CLOTH_W * CLOTH_H, 100.f);
	invmass[0] = 0.f;
	invmass[W - 1] = 0.f;
	invmass[CLOTH_W * (CLOTH_H - 1)] = 0.f;
	invmass[CLOTH_W * CLOTH_H - 1] = 0.f;
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

	ConstraintStorage<RigidBodyConstraint>::Instance.setCpuConstraints(rigidBody.getConstraints());
	ConstraintStorage<DistanceConstraint>::Instance.addStaticConstraints(cloth.getConstraints().first, cloth.getConstraints().second);

	gpuErrchk(cudaFree(dev_curand));
}
