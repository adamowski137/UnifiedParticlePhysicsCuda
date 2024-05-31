#include "Scene_RigidBody.cuh"
#include "../../ResourceManager/ResourceManager.hpp"
#include "../../PhysicsEngine/Particle/ParticleData.cuh"
#include "../../PhysicsEngine/GpuErrorHandling.hpp"
#include "../../PhysicsEngine/Constants.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#define amountOfPoints 576

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
	//particles.setSurfaces({ Surface().init(1, 3, 0, 0), Surface().init(1, 0, 0, 20), Surface().init(-1, 0, 0, 20)});
	particles.setSurfaces({ Surface().init(0, 1, 0, 0)});
	//particles.setSurfaces({ });


	applySceneSetup();

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
	particles.sendDataToVBO(sceneSphere.instancingVBO);
	renderer->drawInstanced(sceneSphere, particles.particleCount());
}

void Scene_RigidBody::reset()
{
	particles.clearConstraints();
	ConstraintStorage<RigidBodyConstraint>::Instance.setCpuConstraints(rigidBody.getConstraints());
}

void Scene_RigidBody::initData(int nParticles,
	float* dev_x, float* dev_y, float* dev_z,
	float* dev_vx, float* dev_vy, float* dev_vz,
	int* dev_SDF_mode, float* dev_SDF_value, float* dev_SDF_normal_x, float* dev_SDF_normal_y, float* dev_SDF_normal_z,
	int* dev_phase, float* dev_invmass)
{
	curandState* dev_curand;
	int threads = 32;
	int blocks = (nParticles + threads - 1) / threads;
	gpuErrchk(cudaMalloc((void**)&dev_curand, nParticles * sizeof(curandState)));

	initializeRandomKern << < blocks, threads >> > (nParticles, dev_curand);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	/*fillRandomKern << <blocks, threads >> > (nParticles - 64, &dev_x[64], dev_curand, -8.f, 8.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	fillRandomKern << <blocks, threads >> > (nParticles - 64, &dev_y[64], dev_curand, 0.f, 30.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	fillRandomKern << <blocks, threads >> > (nParticles - 64, &dev_z[64], dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());*/


	fillRandomKern << <blocks, threads >> > (nParticles, dev_vx, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vy, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vz, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	rigidBody.addRigidBodySquare(dev_x, dev_y, dev_z,
		dev_SDF_mode, dev_SDF_value, dev_SDF_normal_x, dev_SDF_normal_y, dev_SDF_normal_z,
		dev_invmass, 0, 4, 0, 30, 0, dev_phase, 3);
	rigidBody.addRigidBodySquare(dev_x, dev_y, dev_z,
		dev_SDF_mode, dev_SDF_value, dev_SDF_normal_x, dev_SDF_normal_y, dev_SDF_normal_z,
		dev_invmass, 64, 8, 0, 6, 0, dev_phase, 4);

}

