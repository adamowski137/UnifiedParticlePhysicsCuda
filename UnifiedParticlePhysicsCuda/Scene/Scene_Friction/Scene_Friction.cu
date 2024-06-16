#include "Scene_Friction.cuh"
#include "../../ResourceManager/ResourceManager.hpp"

#include "../../PhysicsEngine/Particle/ParticleData.cuh"
#include "../../PhysicsEngine/GpuErrorHandling.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#define A 10
#define H 20
#define N_RIGID_BODY 5
#define N_PARTICLES (A * A * H + N_RIGID_BODY * N_RIGID_BODY * N_RIGID_BODY)
Scene_Friction::Scene_Friction() : Scene(
	ResourceManager::Instance.Shaders["instancedphong"], N_PARTICLES, ANY_CONSTRAINTS_ON | SURFACE_CHECKING_ON | GRID_CHECKING_ON)
{
	std::vector<float> offsets;
	offsets.resize(N_PARTICLES * 3, 0.0f);

	renderer->setSphereScale(0.1f);

	sceneSphere.addInstancing(offsets);
	particles.mapCudaVBO(sceneSphere.instancingVBO);
	particles.setConstraints({ }, 2.f);
	particles.setExternalForces(0.f, -9.81f, 0.f);
	particles.setSurfaces({ Surface().init(0, 1, 0, 0)});

	camera.setPosition(glm::vec3(0, 0, -10));
	applySceneSetup();
}

Scene_Friction::~Scene_Friction()
{
}

void Scene_Friction::update(float dt)
{
	if(!isPaused)	particles.calculateNewPositions(dt);
	this->handleKeys();

	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix());
	renderer->setCameraPosition(camera.getPosition());
	renderer->setLightSourcePosition(glm::vec3(0, 0, -10));

}

void Scene_Friction::draw()
{
	particles.sendDataToVBO(sceneSphere.instancingVBO);
	renderer->drawInstanced(sceneSphere, particles.particleCount());
}

void Scene_Friction::reset()
{
	particles.clearConstraints();
	ConstraintStorage<RigidBodyConstraint>::Instance.setCpuConstraints(rigidBody.getConstraints());
}

void Scene_Friction::initData(int nParticles, float* dev_x, float* dev_y, float* dev_z, float* dev_vx, float* dev_vy, float* dev_vz, int* dev_phase, float* dev_invmass)
{
	curandState* dev_curand;
	int threads = 32;

	int blocks = (nParticles + threads - 1) / threads;
	gpuErrchk(cudaMalloc((void**)&dev_curand, nParticles * sizeof(curandState)));

	initializeRandomKern << < blocks, threads >> > (nParticles, dev_curand);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	auto x_ptr = thrust::device_pointer_cast(dev_x);
	auto y_ptr = thrust::device_pointer_cast(dev_y);
	auto z_ptr = thrust::device_pointer_cast(dev_z);

	for (int x = 0; x < A; x++)
	{
		for (int z = 0; z < A; z++)
		{
			for (int y = 0; y < H; y++)
			{
				x_ptr[(A * H) * x + H * z + y] = x * 2.f;
				y_ptr[(A * H) * x + H * z + y] = 1.f + y * 2.f;
				z_ptr[(A * H) * x + H * z + y] = z * 2.f;
			}
		}
	}
	
	auto invmass_ptr = thrust::device_pointer_cast(dev_invmass);
	thrust::fill(invmass_ptr + A * A * H, invmass_ptr + nParticles, 0.f);


	rigidBody.addRigidBodySquare(dev_x, dev_y, dev_z, dev_invmass, A * A * H, N_RIGID_BODY, -10, 20, -5, dev_phase, 3);
	
	auto vx_ptr = thrust::device_pointer_cast(dev_vx);
	auto vy_ptr = thrust::device_pointer_cast(dev_vy);
	auto vz_ptr = thrust::device_pointer_cast(dev_vz);

	thrust::fill(vx_ptr, vx_ptr + N_PARTICLES, 0.f);
	thrust::fill(vy_ptr, vy_ptr + N_PARTICLES, 0.f);
	thrust::fill(vz_ptr, vz_ptr + N_PARTICLES, 0.f);

	ConstraintStorage<RigidBodyConstraint>::Instance.setCpuConstraints(rigidBody.getConstraints());

	cudaFree(dev_curand);
}
