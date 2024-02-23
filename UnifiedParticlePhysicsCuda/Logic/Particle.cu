#include <glad/glad.h>
#include "Particle.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#define EPS 0.000001

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
	}

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

	gpuErrchk(cudaMalloc((void**)&dev_new_x, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_new_y, amountOfParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_new_z, amountOfParticles * sizeof(float)));
	
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

void ParticleType::calculateNewPositions(float dt)
{
	// predict new positions and update velocities

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
