#include <glad/glad.h>
#include "Particle.cuh"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <memory>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include "../GpuErrorHandling.hpp"
#include "../Constrain/DistanceConstrain/DistanceConstrain.cuh"

#define EPS 0.000005
#define SHMEM_SIZE 1024

__global__ void initializeRandomKern(int amount, curandState* state)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= amount) return;
	curand_init(1234, index, 0, &state[index]);
}

__global__ void fillRandomKern(int amount, float* dst, curandState* state, float min, float max)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= amount) return;
	dst[index] = (max - min) * curand_uniform(&state[index]) + min;
}

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
	float* vx, float* vy, float* vz, float* fc, float invdt)
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

ParticleType::ParticleType(int amount) : amountOfParticles{amount}
{
	blocks = ceilf((float)amountOfParticles / THREADS);
	constrainSolver = std::unique_ptr<ConstrainSolver>{ new ConstrainSolver{amount, 1}};
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
}

void ParticleType::setupDeviceData()
{
	gpuErrchk(cudaMalloc((void**)&dev_curand, amountOfParticles * sizeof(curandState)));
	gpuErrchk(cudaMalloc((void**)&dev_x, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_y, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_z, amountOfParticles * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_new_x, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_new_y, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_new_z, amountOfParticles * sizeof(float)));
	
	gpuErrchk(cudaMalloc((void**)&dev_vx, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_vy, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_vz, amountOfParticles * sizeof(float)));
	
	gpuErrchk(cudaMalloc((void**)&dev_fc, 3 * amountOfParticles * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&dev_invmass, amountOfParticles * sizeof(float)));
	thrust::device_ptr<float> massptr{ dev_invmass };
	thrust::fill(massptr, massptr + amountOfParticles, 1);

	int amountOfConstrains = amountOfParticles;

	initializeRandomKern << < blocks, THREADS >> > (amountOfParticles, dev_curand);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, THREADS >> > (amountOfParticles, dev_x, dev_curand, -30.f, 30.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, THREADS >> > (amountOfParticles, dev_y, dev_curand, -30.f, 30.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	fillRandomKern << <blocks, THREADS >> > (amountOfParticles, dev_z, dev_curand, -30.f, 30.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, THREADS >> > (amountOfParticles, dev_vx, dev_curand, -5.f, 5.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	//fillRandomKern << <blocks, THREADS >> > (amountOfParticles, dev_vy, dev_curand, -5.f, 5.f);
	//gpuErrchk(cudaGetLastError());
	//gpuErrchk(cudaDeviceSynchronize());
	
	fillRandomKern << <blocks, THREADS >> > (amountOfParticles, dev_vz, dev_curand, -1.f, 1.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());*/
}

void ParticleType::renderData(unsigned int vbo)
{
	float* dst;
	cudaGLMapBufferObject((void**)&dst, vbo);

	copyToVBOKernel <<<blocks, THREADS>>>(amountOfParticles, dev_x, dev_y, dev_z, dst);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	cudaGLUnmapBufferObject(vbo);
}

void ParticleType::calculateNewPositions(float dt)
{
	cudaMemset(dev_fc, 0, 3 * amountOfParticles * sizeof(float));
	// predict new positions and update velocities
	fextx = 0.0f;
	//fexty = -9.81f;
	fexty = 0.0f;
	fextz = 0.0f;

	float dvx = fextx * dt;
	float dvy = fexty * dt;
	float dvz = fextz * dt;

	predictPositionsKern << <blocks, THREADS >> > (
		amountOfParticles,
		dev_x, dev_y, dev_z,
		dev_new_x, dev_new_y, dev_new_z,
		dev_vx, dev_vy, dev_vz,
		dvx, dvy, dvz, dt
		);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// find neighboring particles and solid contacts ??

	// todo implement grid (predicted positions)

	// stabilization iterations

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// todo solve contact constrains
	// update predicted position and current positions

	// solve iterations

	constrainSolver.get()->calculateForces(
		dev_x, dev_y, dev_z,
		dev_new_x, dev_new_y, dev_new_z,
		dev_vx, dev_vy, dev_vz,
		dev_invmass, dev_fc, dt);

	// todo solve every constraint group 
	// update predicted position
	applyChangesKern << <blocks, THREADS >> > (
		amountOfParticles,
		dev_x, dev_y, dev_z,
		dev_new_x, dev_new_y, dev_new_z,
		dev_vx, dev_vy, dev_vz, dev_fc,
		1/dt
		);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void ParticleType::mapCudaVBO(unsigned int vbo)
{
	cudaGLRegisterBufferObject(vbo);
}