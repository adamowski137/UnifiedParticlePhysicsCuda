#include "TestScene.cuh"
#include "../../ResourceManager/ResourceManager.hpp"

#include "../../PhysicsEngine/Particle/ParticleData.cuh"
#include "../../PhysicsEngine/GpuErrorHandling.hpp"
#include <curand.h>
#include <curand_kernel.h>

TestScene::TestScene(int n) : Scene(ResourceManager::Instance.Shaders["instancedphong"], n,
	ANY_CONSTRAINTS_ON | GRID_CHECKING_ON | SURFACE_CHECKING_ON)
{

	std::vector<float> offsets;
	offsets.resize(n * 3, 0.0f);

	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setConstraints({ }, 2.f);
	particles.setExternalForces(0.f, -9.81f, 0.f);
	particles.setSurfaces({ Surface().init(0, 1, 0, 0), Surface().init(1, 0, 0, 20), Surface().init(-1, 0, 0, 20), Surface().init(0, 0, 1, 20), Surface().init(0, 0, -1, 20)});

	camera.setPosition(glm::vec3(0, 0, -10));

	applySceneSetup();
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
	particles.sendDataToVBO(sceneSphere.instancingVBO);
	renderer->drawInstanced(sceneSphere, particles.particleCount());
}

void TestScene::initData(int nParticles, float* dev_x, float* dev_y, float* dev_z, float* dev_vx, float* dev_vy, float* dev_vz, int* dev_phase, float* dev_invmass)
{
	//gpuErrchk(cudaMemset(dev_x, 0, sizeof(float) * 3 * nParticles));
	//gpuErrchk(cudaMemset(dev_y, 0, sizeof(float) * 3 * nParticles));
	//gpuErrchk(cudaMemset(dev_z, 0, sizeof(float) * 3 * nParticles));
	//gpuErrchk(cudaMemset(dev_vx, 0, sizeof(float) * 3 * nParticles));
	//gpuErrchk(cudaMemset(dev_vy, 0, sizeof(float) * 3 * nParticles));
	//gpuErrchk(cudaMemset(dev_vz, 0, sizeof(float) * 3 * nParticles));
	curandState* dev_curand;
	int threads = 32;

	int blocks = (nParticles + threads - 1) / threads;
	gpuErrchk(cudaMalloc((void**)&dev_curand, nParticles * sizeof(curandState)));

	initializeRandomKern << < blocks, threads >> > (nParticles, dev_curand);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_x, dev_curand, -10.f, 10.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_y, dev_curand, 8.f, 78.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_z, dev_curand, -10.f, 10.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vx, dev_curand, -10.f, 10.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vy, dev_curand, -10.f, 10.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vz, dev_curand, -10.f, 10.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	//TEST FOR 2 PARTICLES
	//float tmp[2] = { 5.f, -10.f };
	//gpuErrchk(cudaMemcpy(dev_x, &tmp[0], sizeof(float) * 2, cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(dev_y, &tmp[0], sizeof(float) * 2, cudaMemcpyHostToDevice));

	//gpuErrchk(cudaGetLastError());
	//gpuErrchk(cudaDeviceSynchronize());

	//tmp[0] = -5.f;
	//tmp[1] = 5.f;
	//cudaMemcpy(dev_vx, &tmp[0], sizeof(float) * 2, cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_vy, &tmp[0], sizeof(float) * 2, cudaMemcpyHostToDevice);

	//gpuErrchk(cudaGetLastError());
	//gpuErrchk(cudaDeviceSynchronize());
	cudaFree(dev_curand);
}
