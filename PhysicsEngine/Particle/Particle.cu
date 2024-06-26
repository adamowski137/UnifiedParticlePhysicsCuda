#include <memory>
#include <cstdlib>
#include <glad/glad.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <device_launch_parameters.h>
#include "Particle.cuh"
#include "../Constants.hpp"
#include "ParticleData.cuh"
#include "../GpuErrorHandling.hpp"
#include "../Constraint/DistanceConstraint/DistanceConstraint.cuh"
#include "../Math/ConstraintSolver/LinearSystemConstraintSolver/LinearSystemConstraintSolver.cuh"
#include "../Math/ConstraintSolver/DirectConstraintSolver/DirectConstraintSolver.cuh"
#include "../Math/ConstraintSolver/DirectConstraintSolverCPU/DirectConstraintSolverCPU.cuh"

#define EPS 0.0000001


__global__ void copyToVBOKernel(int n, int offset, float* x, float* y, float* z, float* dst)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) return;
	dst[3 * index + 0] = x[index + offset];
	dst[3 * index + 1] = y[index + offset];
	dst[3 * index + 2] = z[index + offset];
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
	float dt, float* invmass
)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= amount) return;

	// update velocities
	if (invmass[index] > 0)
	{

		vx[index] += dvx;
		vy[index] += dvy;
		vz[index] += dvz;
	}

	// predict new position - not the actual new positions
	new_x[index] = x[index] + dt * vx[index];
	new_y[index] = y[index] + dt * vy[index];
	new_z[index] = z[index] + dt * vz[index];

	// apply mass scaling??
}

__global__ void applyChangesKern(int amount,
	float* x, float* y, float* z,
	float* new_x, float* new_y, float* new_z,
	float* vx, float* vy, float* vz, float invdt, float d, int* mode)
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

ParticleType::ParticleType(int amount, int mode) : nParticles{ amount }, mode{ mode }
{
	blocks = ceilf((float)nParticles / THREADS);
	//constraintSolver = std::unique_ptr<ConstraintSolver>{ new LinearSystemConstraintSolver{amount} };
	constraintSolver = std::unique_ptr<ConstraintSolver>{ new DirectConstraintSolver{amount} };
	//constraintSolver = std::unique_ptr<ConstraintSolver>{ new DirectConstraintSolverCPU{amount} };
	collisionGrid = std::unique_ptr<CollisionGrid>{ new CollisionGrid{amount} };
	surfaceCollisionFinder = std::unique_ptr<SurfaceCollisionFinder>{ new SurfaceCollisionFinder{ { } , amount} };
	allocateDeviceData();
	setupDeviceData();
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
	gpuErrchk(cudaFree(dev_phase));
}

void ParticleType::setupDeviceData()
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

	cudaFree(dev_curand);
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


	gpuErrchk(cudaMalloc((void**)&dev_phase, nParticles * sizeof(int)));
	// by default every particle is in a separate group
	thrust::device_ptr<int> phaseptr{ dev_phase };
	thrust::sequence(phaseptr, phaseptr + nParticles, 1);



	gpuErrchk(cudaMalloc((void**)&dev_fc, 3 * nParticles * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_invmass, nParticles * sizeof(float)));
	thrust::device_ptr<float> massptr{ dev_invmass };
	thrust::fill(massptr, massptr + nParticles, 1);


}

void ParticleType::sendDataToVBO(unsigned int vbo, int startIdx, int n)
{
	float* dst;
	cudaGLMapBufferObject((void**)&dst, vbo);

	int n_bound_blocks = (n + THREADS - 1) / THREADS;

	copyToVBOKernel << <n_bound_blocks, THREADS >> > (n, startIdx, dev_x, dev_y, dev_z, dst);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaGLUnmapBufferObject(vbo);
}

void ParticleType::sendDataToVBO(unsigned int vbo)
{
	sendDataToVBO(vbo, 0, this->nParticles);
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
		dvx, dvy, dvz, dt, dev_invmass
		);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//find neighboring particles and solid contacts ??

   //if (mode & GRID_CHECKING_ON)
   //	collisionGrid->findAndUpdateCollisions(dev_x, dev_y, dev_z, dev_phase, nParticles);

   //if (mode & SURFACE_CHECKING_ON)
   //	surfaceCollisionFinder->findAndUpdateCollisions(nParticles, dev_x, dev_y, dev_z);

   //if (mode & ANY_CONSTRAINTS_ON)
   //	constraintSolver->calculateStabilisationForces(dev_x, dev_y,dev_z, dev_phase, dev_new_x, dev_new_y, dev_new_z, dev_invmass, dt, 1);

   // solve iterations
	if (mode & GRID_CHECKING_ON)
		collisionGrid->findAndUpdateCollisions(dev_new_x, dev_new_y, dev_new_z, dev_phase, nParticles);

	if (mode & SURFACE_CHECKING_ON)
		surfaceCollisionFinder->findAndUpdateCollisions(nParticles, dev_new_x, dev_new_y, dev_new_z);

	if (mode & ANY_CONSTRAINTS_ON)
		constraintSolver->calculateForces(dev_x, dev_y, dev_z, dev_phase, dev_new_x, dev_new_y, dev_new_z, dev_invmass, dt, 5);

	// todo solve every constraint group 
	// update predicted position
	applyChangesKern << <blocks, THREADS >> > (
		nParticles,
		dev_x, dev_y, dev_z,
		dev_new_x, dev_new_y, dev_new_z,
		dev_vx, dev_vy, dev_vz,
		1 / dt, dt, dev_phase
		);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void ParticleType::setConstraints(std::vector<std::pair<int, int>> pairs, float d)
{
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

void ParticleType::clearConstraints()
{
	ConstraintStorage<RigidBodyConstraint>::Instance.clearConstraints();
	ConstraintStorage<DistanceConstraint>::Instance.clearConstraints();
	ConstraintStorage<SurfaceConstraint>::Instance.clearConstraints();
}


void ParticleType::mapCudaVBO(unsigned int vbo)
{
	cudaGLRegisterBufferObject(vbo);
}

