#include <glad/glad.h>
#include "Particle.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
__global__ void copyKernel(int amount, float* src, float* dst)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= amount) return;
	dst[index] = src[index];
}

__global__ void setDiagonalMatrix(int amount, float* src, float* dst)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= amount) return;
	dst[amount * index + index] = src[index];
}

ParticleType::ParticleType(int amount, float mass) : amountOfParticles{amount}
{
	blocks = ceilf((float)amountOfParticles / THREADS);

	setupDeviceData();
}

ParticleType::~ParticleType()
{
	gpuErrchk(cudaFree(dev_x));
	gpuErrchk(cudaFree(dev_y));
	gpuErrchk(cudaFree(dev_z));
	gpuErrchk(cudaFree(dev_vx));
	gpuErrchk(cudaFree(dev_vy));
	gpuErrchk(cudaFree(dev_vz));
}

void ParticleType::setupDeviceData()
{
	gpuErrchk(cudaMalloc((void**)&dev_x, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_y, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_z, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_vx, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_vy, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_vz, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_invmass, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_invM, amountOfParticles * amountOfParticles * sizeof(float)));

	setDiagonalMatrix << <THREADS, blocks >> > (amountOfParticles, dev_invmass, dev_invM);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void ParticleType::setupShaderData()
{
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vboSphere);
	glGenBuffers(1, &vbox);
	glGenBuffers(1, &vboy);
	glGenBuffers(1, &vboz);
	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbox);
	glBufferData(GL_ARRAY_BUFFER, (amountOfParticles) * sizeof(GLfloat), nullptr, GL_STREAM_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer((GLuint)0, 1, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, vboy);
	glBufferData(GL_ARRAY_BUFFER, (amountOfParticles) * sizeof(GLfloat), nullptr, GL_STREAM_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer((GLuint)1, 1, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, vboy);
	glBufferData(GL_ARRAY_BUFFER, (amountOfParticles) * sizeof(GLfloat), nullptr, GL_STREAM_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer((GLuint)1, 1, GL_FLOAT, GL_FALSE, 0, 0);

	gpuErrchk(cudaGLRegisterBufferObject(vboSphere));
	gpuErrchk(cudaGLRegisterBufferObject(vbox));
	gpuErrchk(cudaGLRegisterBufferObject(vboy));
	gpuErrchk(cudaGLRegisterBufferObject(vboz));
}

void ParticleType::renderData()
{
	float* x, * y, * z;
	cudaGLMapBufferObject((void**)&x, vbox);
	cudaGLMapBufferObject((void**)&y, vboy);
	cudaGLMapBufferObject((void**)&z, vboz);

	copyKernel<<<blocks, THREADS>>>(amountOfParticles, dev_x, x);
	copyKernel<<<blocks, THREADS>>>(amountOfParticles, dev_y, y);
	copyKernel<<<blocks, THREADS>>>(amountOfParticles, dev_z, z);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	cudaGLUnmapBufferObject(vbox);
	cudaGLUnmapBufferObject(vboy);
	cudaGLUnmapBufferObject(vboz);

	glBindVertexArray(vao);
}