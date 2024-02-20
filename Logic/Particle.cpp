#include "glad/glad.h"
#include "Particle.hpp"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <cuda_gl_interop.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}



ParticleType::ParticleType(int amount, float mass) : amountOfParticles{amount}, invmass{mass}
{
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
}

void ParticleType::setupShaderData()
{
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);

	gpuErrchk(cudaGLRegisterBufferObject(vbo));
}
