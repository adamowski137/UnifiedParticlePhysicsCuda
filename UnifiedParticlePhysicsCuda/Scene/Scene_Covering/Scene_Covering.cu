#include "Scene_Covering.cuh"
#include "../../ResourceManager/ResourceManager.hpp"

#include "../../PhysicsEngine/Particle/ParticleData.cuh"
#include "../../PhysicsEngine/GpuErrorHandling.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#define CLOTH_W 30
#define CLOTH_H 30
#define N_RIGID_BODY 5
#define NUM_PARTICLES (CLOTH_W * CLOTH_H + N_RIGID_BODY * N_RIGID_BODY * N_RIGID_BODY)

Scene_Covering::Scene_Covering() :
	Scene(ResourceManager::Instance.Shaders["instancedphong"], NUM_PARTICLES, ANY_CONSTRAINTS_ON | SURFACE_CHECKING_ON | GRID_CHECKING_ON)
{
	std::vector<float> offsets;
	offsets.resize(NUM_PARTICLES * 3, 0.0f);

	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setExternalForces(0.f, -9.81f, 0.f);
	particles.setSurfaces({ Surface().init(0, 1, 0, 0)});

	camera.setPosition(glm::vec3(0, 0, -10));

	applySceneSetup();
	ConstraintStorage<DistanceConstraint>::Instance.addStaticConstraints(cloth.getConstraints().first, cloth.getConstraints().second);
	ConstraintStorage<RigidBodyConstraint>::Instance.setCpuConstraints(rigidBody.getConstraints());
}

Scene_Covering::~Scene_Covering()
{
}

void Scene_Covering::update(float dt)
{
	particles.calculateNewPositions(dt);
	this->handleKeys();

	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix());
	renderer->setCameraPosition(camera.getPosition());
	renderer->setLightSourcePosition(glm::vec3(0, 0, -10));
}

void Scene_Covering::draw()
{
	particles.sendDataToVBO(sceneSphere.instancingVBO);
	renderer->drawInstanced(sceneSphere, particles.particleCount());
}

void Scene_Covering::reset()
{
	particles.clearConstraints();
	ConstraintStorage<RigidBodyConstraint>::Instance.setCpuConstraints(rigidBody.getConstraints());
	ConstraintStorage<DistanceConstraint>::Instance.addStaticConstraints(cloth.getConstraints().first, cloth.getConstraints().second);
}

void Scene_Covering::initData(int nParticles, float* dev_x, float* dev_y, float* dev_z, float* dev_vx, float* dev_vy, float* dev_vz, int* dev_phase, float* dev_invmass)
{
	curandState* dev_curand;
	int threads = 32;

	int blocks = (nParticles + threads - 1) / threads;
	gpuErrchk(cudaMalloc((void**)&dev_curand, nParticles * sizeof(curandState)));

	initializeRandomKern << < blocks, threads >> > (nParticles, dev_curand);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	float d = 1.f;
	int W = CLOTH_W;
	int H = CLOTH_H;
	rigidBody.addRigidBodySquare(dev_x, dev_y, dev_z, dev_invmass, CLOTH_W * CLOTH_H, N_RIGID_BODY, -10, 1, -5, dev_phase, 3);
	Cloth::initClothSimulation_simple(cloth, H, W, d, -d * W / 2.f, 30.f, 15.f, dev_x, dev_y, dev_z, dev_phase, ClothOrientation::XZ_PLANE);

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
