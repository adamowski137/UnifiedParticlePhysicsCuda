#include "Scene_RigidBody.cuh"
#include "../../ResourceManager/ResourceManager.hpp"
#include "../../PhysicsEngine/Particle/ParticleData.cuh"
#include "../../PhysicsEngine/GpuErrorHandling.hpp"
#include "../../PhysicsEngine/Constants.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#define amountOfPoints 64

Scene_RigidBody::Scene_RigidBody() : Scene(
	ResourceManager::Instance.Shaders["instancedphong"], amountOfPoints, ANY_CONSTRAINTS_ON | SURFACE_CHECKING_ON | GRID_CHECKING_ON)
{
	std::vector<float> offsets;
	offsets.resize(amountOfPoints * 3, 0.0f);

	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setConstraints({ }, 2.f);
	particles.setExternalForces(0.f, -9.81f, 0.f);
	particles.setSurfaces({ Surface().init(0, 1, 0, 0), Surface().init(1, 0, 0, 20), Surface().init(-1, 0, 0, 20)});
	applySceneSetup();
	std::vector<int> points;
	for (int i = 0; i < 64; i++)
	{
		points.push_back(i);
	}
	particles.setRigidBodyConstraint(points);

	camera.setPosition(glm::vec3(0, 0, -10));
}

Scene_RigidBody::~Scene_RigidBody()
{
}

void Scene_RigidBody::update(float dt)
{
	particles.calculateNewPositions(dt);
	this->handleKeys();

	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix());
	renderer->setCameraPosition(camera.getPosition());
	renderer->setLightSourcePosition(glm::vec3(0, 0, -10));

}

void Scene_RigidBody::draw()
{
	particles.renderData(sceneSphere.instancingVBO);
	renderer->drawInstanced(sceneSphere, particles.particleCount());
}

void Scene_RigidBody::reset()
{
	//std::vector<float> offsets;
	//offsets.resize(amountOfPoints * 3, 0.0f);

	//renderer->setSphereScale(0.1f);

	//sceneSphere.addInstancing(offsets);
	//particles.mapCudaVBO(sceneSphere.instancingVBO);
	//particles.setConstraints({ }, 2.f);
	//particles.setExternalForces(0.f, -9.81f, 0.f);
	//particles.setSurfaces({ Surface().init(0, 1, 0, 0), Surface().init(1, 0, 0, 20), Surface().init(-1, 0, 0, 20) });
	//std::vector<int> points;
	//for (int i = 0; i < 64; i++)
	//{
	//	points.push_back(i);
	//}
	//particles.setRigidBodyConstraint(points);

	//camera.setPosition(glm::vec3(0, 0, -10));
}

__global__ void initializePositionsKern(int nParticles, float* x, float* y, float* z, int dim)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;

	int xIndex = index % dim;
	int yIndex = 6 + (index / dim) % dim;
	int zIndex = index / (dim * dim);

	x[index] = xIndex * PARTICLERADIUS * 2;
	y[index] = yIndex * PARTICLERADIUS * 2;
	z[index] = zIndex * PARTICLERADIUS * 2;
}

void Scene_RigidBody::initData(int nParticles, float* dev_x, float* dev_y, float* dev_z, float* dev_vx, float* dev_vy, float* dev_vz, int* dev_phase, float* dev_invmass)
{
	curandState* dev_curand;
	int threads = 32;
	int blocks = (nParticles + threads - 1) / threads;
	gpuErrchk(cudaMalloc((void**)&dev_curand, nParticles * sizeof(curandState)));

	initializeRandomKern << < blocks, threads >> > (nParticles, dev_curand);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	initializePositionsKern << <blocks, threads >> > (nParticles, dev_x, dev_y, dev_z, 4);

	fillRandomKern << <blocks, threads >> > (nParticles - 64, &dev_x[64], dev_curand, -8.f, 8.f);
	fillRandomKern << <blocks, threads >> > (nParticles - 64, &dev_y[64], dev_curand, 8.f, 21.f);
	fillRandomKern << <blocks, threads >> > (nParticles - 64, &dev_z[64], dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	fillRandomKern << <blocks, threads >> > (nParticles, dev_vx, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vy, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

