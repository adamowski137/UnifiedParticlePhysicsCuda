#include <memory>
#include <cstdlib>
#include <glad/glad.h>
#include <thrust/fill.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <device_launch_parameters.h>
#include "Particle.cuh"
#include "../Constants.hpp"
#include "ParticleData.cuh"
#include "../GpuErrorHandling.hpp"
#include "../Constraint/DistanceConstraint/DistanceConstraint.cuh"

#define EPS 0.000001


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
	allocateDeviceData();
	setupDeviceData(setDataFunction);
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

	gpuErrchk(cudaMalloc((void**)&dev_dx, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_dy, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_dz, nParticles * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_new_x, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_new_y, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_new_z, nParticles * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_vx, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_vy, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_vz, nParticles * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_invmass, nParticles * sizeof(float)));

	gpuErrchk(cudaMemset(dev_dx, 0, nParticles * sizeof(float)));
	gpuErrchk(cudaMemset(dev_dy, 0, nParticles * sizeof(float)));
	gpuErrchk(cudaMemset(dev_dz, 0, nParticles * sizeof(float)));

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


	// stabilization iterations

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// todo solve contact constrains
	// update predicted position and current positions

	// solve iterations
	if(mode & ANY_CONSTRAINTS_ON)
		constraintSolver->calculateForces(dev_x, dev_y, dev_z, dev_new_x, dev_new_y, dev_new_z, dev_dx, dev_dy, dev_dz, dev_vx, dev_vy, dev_vz, dev_invmass, dt, mode);

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
	this->constraintSolver->setSurfaces(surfaces, nParticles);
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