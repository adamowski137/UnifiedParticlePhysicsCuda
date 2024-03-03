#include <glad/glad.h>
#include "Particle.cuh"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <memory>
#include "../GpuErrorHandling.hpp"
#include "../Constrain/DistanceConstrain/DistanceConstrain.cuh"

#define EPS 0.000001
#define SHMEM_SIZE 1024

__global__ void matrixMulKern(const float* a, const float* b, float* c, int N, int K) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float s_a[SHMEM_SIZE];
	__shared__ float s_b[SHMEM_SIZE];

	float tmp = 0;

	for (int i = 0; i < K; i += blockDim.x)
	{

		s_a[threadIdx.y * blockDim.x + threadIdx.x] = 0;
		s_b[threadIdx.y * blockDim.x + threadIdx.x] = 0;
		__syncthreads();


		if (row < N && col < N)
		{
			if (i + threadIdx.x < K)
				s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i + threadIdx.x];
			if (i + threadIdx.y < K)
				s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];
		}
		__syncthreads();

		if (row < N && col < N)
		{
			for (int j = 0; j < blockDim.x; j++) {
				tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
			}
		}
		__syncthreads();
	}

	if (row < N && col < N)
		c[row * N + col] = tmp;


}


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
	float* vx, float* vy, float* vz, float invdt)
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
	gpuErrchk(cudaFree(dev_invM));

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
	
	gpuErrchk(cudaMalloc((void**)&dev_invmass, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_invM, amountOfParticles * amountOfParticles * sizeof(float)));

	int amountOfConstrains = amountOfParticles;

	gpuErrchk(cudaMalloc((void**)&dev_jacobian, sizeof(float) * 3 * amountOfParticles * amountOfConstrains));
	gpuErrchk(cudaMemset(dev_jacobian, 0, sizeof(float) * 3 * amountOfParticles * amountOfConstrains));

	setDiagonalMatrix << <THREADS, blocks >> > (amountOfParticles, dev_invmass, dev_invM);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	initializeRandomKern << <THREADS, blocks >> > (amountOfParticles, dev_curand);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <THREADS, blocks >> > (amountOfParticles, dev_x, dev_curand, -100.f, 100.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <THREADS, blocks >> > (amountOfParticles, dev_y, dev_curand, -100.f, 100.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	fillRandomKern << <THREADS, blocks >> > (amountOfParticles, dev_z, dev_curand, -100.f, 100.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <THREADS, blocks >> > (amountOfParticles, dev_vx, dev_curand, -100.f, 100.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <THREADS, blocks >> > (amountOfParticles, dev_vy, dev_curand, -100.f, 100.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <THREADS, blocks >> > (amountOfParticles, dev_vz, dev_curand, -100.f, 100.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
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
	// predict new positions and update velocities
	fextx = 0.0f;
	fexty = -9.81f;
	fextz = 0.0f;

	float dvx = fextx * dt;
	float dvy = fexty * dt;
	float dvz = fextz * dt;

	predictPositionsKern << <THREADS, blocks >> > (
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

	// todo solve every constraint group 
	// update predicted position
	applyChangesKern << <THREADS, blocks >> > (
		amountOfParticles,
		dev_x, dev_y, dev_z,
		dev_new_x, dev_new_y, dev_new_z,
		dev_vx, dev_vy, dev_vz,
		1/dt
		);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void ParticleType::mapCudaVBO(unsigned int vbo)
{
	cudaGLRegisterBufferObject(vbo);
}