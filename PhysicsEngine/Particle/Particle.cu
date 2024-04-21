#include <memory>
#include <cstdlib>
#include <glad/glad.h>
#include <thrust/fill.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <device_launch_parameters.h>
#include <chrono>
#include "Particle.cuh"
#include "../Constants.hpp"
#include "ParticleData.cuh"
#include "../GpuErrorHandling.hpp"
#include "../Constraint/DistanceConstraint/DistanceConstraint.cuh"
#include "../RigidBody/RigidBody.cuh"

#define EPS 0.00001


__global__ void copyToVBOKernel(int amount, float* x, float* y, float* z, float* dst)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= amount) return;
	dst[3 * index + 0] = x[index];
	dst[3 * index + 1] = y[index];
	dst[3 * index + 2] = z[index];
}

__global__ void setDiagonalMatrix(int amount, float* src, float* dst)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= amount) return;
	dst[amount * index + index] = src[index];
}

__global__ void predictPositionsKern(int amount,
	float* x, float* y, float* z,
	float* new_x, float* new_y, float* new_z,
	float* vx, float* vy, float* vz,
	float dvx, float dvy, float dvz,
	float dt
)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= amount) return;

	// update velocities
	vx[index] += dvx;
	vy[index] += dvy;
	vz[index] += dvz;

	// predict new position - not the actual new positions
	new_x[index] = x[index] + dt * vx[index];
	new_y[index] = y[index] + dt * vy[index];
	new_z[index] = z[index] + dt * vz[index];

	// apply mass scaling??
}

__global__ void applyChangesKern(int amount,
	float* x, float* y, float* z,
	float* new_x, float* new_y, float* new_z,
	float* vx, float* vy, float* vz, float invdt, float dt)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index >= amount) return;

	float changeX = (new_x[index] - x[index]);
	float changeY = (new_y[index] - y[index]);
	float changeZ = (new_z[index] - z[index]);

	// update velocity
	vx[index] = invdt * (changeX);
	vy[index] = invdt * (changeY);
	vz[index] = invdt * (changeZ);

	// advect diffuse particles ??

	// apply internal forces

	// update position or apply sleeping

	float changeSQ = changeX * changeX + changeY * changeY + changeZ * changeZ;
	if (changeSQ > EPS)
	{
		x[index] = new_x[index];
		y[index] = new_y[index];
		z[index] = new_z[index];
	}

}

ParticleType::ParticleType(int amount, int mode,
	void(*setDataFunction)(int, float*, float*, float*, float*, float*, float*)) : nParticles{ amount }, mode{ mode }
{
	blocks = ceilf((float)nParticles / THREADS);
	constraintSolver = std::unique_ptr<ConstraintSolver>{ new ConstraintSolver{amount} };
	collisionGrid = std::unique_ptr<CollisionGrid>{ new CollisionGrid{amount} };
	surfaceCollisionFinder = std::unique_ptr<SurfaceCollisionFinder>{ new SurfaceCollisionFinder{ { } , amount} };
	allocateDeviceData();
	setupDeviceData(setDataFunction);

	std::vector<int> points{};
	points.push_back(0);
	points.push_back(1);
	RigidBody{ points, dev_x, dev_y, dev_z, dev_invmass };
}

ParticleType::~ParticleType()
{
	gpuErrchk(cudaFree(dev_x));
	gpuErrchk(cudaFree(dev_y));
	gpuErrchk(cudaFree(dev_z));
	gpuErrchk(cudaFree(dev_new_x));
	gpuErrchk(cudaFree(dev_new_y));
	gpuErrchk(cudaFree(dev_new_z));
	gpuErrchk(cudaFree(dev_vx));
	gpuErrchk(cudaFree(dev_vy));
	gpuErrchk(cudaFree(dev_vz));
	gpuErrchk(cudaFree(dev_invmass));
}

void ParticleType::setupDeviceData(void(*setDataFunction)(int, float*, float*, float*, float*, float*, float*))
{
	if (setDataFunction != nullptr)
	{
		setDataFunction(nParticles, dev_x, dev_y, dev_z, dev_vx, dev_vy, dev_vz);
	}
	else
	{
		gpuErrchk(cudaMalloc((void**)&dev_curand, nParticles * sizeof(curandState)));

		initializeRandomKern << < blocks, THREADS >> > (nParticles, dev_curand);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());

		fillRandomKern << <blocks, THREADS >> > (nParticles, dev_x, dev_curand, -10.f, 10.f);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());

		fillRandomKern << <blocks, THREADS >> > (nParticles, dev_y, dev_curand, 5.f, 15.f);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());

		fillRandomKern << <blocks, THREADS >> > (nParticles, dev_z, dev_curand, 0.f, 0.f);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());

		fillRandomKern << <blocks, THREADS >> > (nParticles, dev_vx, dev_curand, -1.f, 1.f);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());

		fillRandomKern << <blocks, THREADS >> > (nParticles, dev_vy, dev_curand, 0.f, 0.f);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());

		//fillRandomKern << <blocks, THREADS >> > (nParticles, dev_vz, dev_curand, -5.f, 5.f);
		//gpuErrchk(cudaGetLastError());
		//gpuErrchk(cudaDeviceSynchronize());
	}
}

void ParticleType::allocateDeviceData()
{
	gpuErrchk(cudaMalloc((void**)&dev_curand, nParticles * sizeof(curandState)));
	gpuErrchk(cudaMalloc((void**)&dev_x, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_y, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_z, nParticles * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_new_x, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_new_y, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_new_z, nParticles * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_vx, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_vy, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_vz, nParticles * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_collisions, nParticles * sizeof(List)));
	gpuErrchk(cudaMalloc((void**)&dev_sums, nParticles * sizeof(int)));

	gpuErrchk(cudaMalloc((void**)&dev_fc, 3 * nParticles * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_invmass, nParticles * sizeof(float)));
	thrust::device_ptr<float> massptr{ dev_invmass };
	thrust::fill(massptr, massptr + nParticles, 1);


}

void ParticleType::renderData(unsigned int vbo)
{
	float* dst;
	cudaGLMapBufferObject((void**)&dst, vbo);

	copyToVBOKernel << <blocks, THREADS >> > (nParticles, dev_x, dev_y, dev_z, dst);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaGLUnmapBufferObject(vbo);
}

void ParticleType::calculateNewPositions(float dt)
{
	float dvx = fextx * dt;
	float dvy = fexty * dt;
	float dvz = fextz * dt;

	predictPositionsKern << <blocks, THREADS >> > (
		nParticles,
		dev_x, dev_y, dev_z,
		dev_new_x, dev_new_y, dev_new_z,
		dev_vx, dev_vy, dev_vz,
		dvx, dvy, dvz, dt
		);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// find neighboring particles and solid contacts ??

	if(mode & GRID_CHECKING_ON)
		collisionGrid->findCollisions(dev_x, dev_y, dev_z, nParticles, dev_sums, dev_collisions);
	
	auto surfaceCollisionData = (mode & SURFACE_CHECKING_ON) 
		? surfaceCollisionFinder->findAndUpdateCollisions(nParticles, dev_x, dev_y, dev_z) 
		: std::make_pair((SurfaceConstraint*)0, 0);
	
	if (mode & GRID_CHECKING_ON)
		constraintSolver->addDynamicConstraints(dev_collisions, dev_sums, PARTICLERADIUS, ConstraintLimitType::GEQ);
	if (mode & SURFACE_CHECKING_ON)
		constraintSolver->addSurfaceConstraints(surfaceCollisionData.first, surfaceCollisionData.second);
	
	// stabilization iterations
	if (mode & ANY_CONSTRAINTS_ON)
		constraintSolver->calculateStabilisationForces(dev_x, dev_y, dev_z, dev_new_x, dev_new_y, dev_new_z, dev_invmass, dt, 5);

	// solve iterations
	if (mode & GRID_CHECKING_ON)
		collisionGrid->findCollisions(dev_new_x, dev_new_y, dev_new_z, nParticles, dev_sums, dev_collisions);
	
	surfaceCollisionData = (mode & SURFACE_CHECKING_ON)
		? surfaceCollisionFinder->findAndUpdateCollisions(nParticles, dev_new_x, dev_new_y, dev_new_z)
		: std::make_pair((SurfaceConstraint*)0, 0);
	
	if(mode & GRID_CHECKING_ON)
		constraintSolver->addDynamicConstraints(dev_collisions, dev_sums, PARTICLERADIUS, ConstraintLimitType::GEQ);
	if(mode & SURFACE_CHECKING_ON)
		constraintSolver->addSurfaceConstraints(surfaceCollisionData.first, surfaceCollisionData.second);
	if(mode & ANY_CONSTRAINTS_ON)
		constraintSolver->calculateForces(dev_new_x, dev_new_y, dev_new_z, dev_invmass, dt, 50);



	// todo solve every constraint group 
	// update predicted position
	applyChangesKern << <blocks, THREADS >> > (
		nParticles,
		dev_x, dev_y, dev_z,
		dev_new_x, dev_new_y, dev_new_z,
		dev_vx, dev_vy, dev_vz,
		1 / dt, dt
		);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void ParticleType::setConstraints(std::vector<std::pair<int, int>> pairs, float d)
{
	this->constraintSolver->setStaticConstraints(pairs, d);
}

void ParticleType::setSurfaces(std::vector<Surface> surfaces)
{
	this->surfaceCollisionFinder->setSurfaces(surfaces, nParticles);
}

void ParticleType::setExternalForces(float fx, float fy, float fz)
{
	fextx = fx;
	fexty = fy;
	fextz = fz;
}

void ParticleType::mapCudaVBO(unsigned int vbo)
{
	cudaGLRegisterBufferObject(vbo);
}