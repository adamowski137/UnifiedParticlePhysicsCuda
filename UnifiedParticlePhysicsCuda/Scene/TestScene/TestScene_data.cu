#include "TestScene_data.cuh"
#include "../../PhysicsEngine/Particle/ParticleData.cuh"
#include "../../PhysicsEngine/GpuErrorHandling.hpp"
#include <cuda_runtime.h>


void initData_TestScene(int nParticles,
	float* dev_x, float* dev_y, float* dev_z,
	float* dev_vx, float* dev_vy, float* dev_vz, int* mode)
{
	//gpuErrchk(cudaMemset(dev_x, 0, sizeof(float) * 3 * nParticles));
	//gpuErrchk(cudaMemset(dev_y, 0, sizeof(float) * 3 * nParticles));
	//gpuErrchk(cudaMemset(dev_z, 0, sizeof(float) * 3 * nParticles));
	//gpuErrchk(cudaMemset(dev_vx, 0, sizeof(float) * 3 * nParticles));
	//gpuErrchk(cudaMemset(dev_vy, 0, sizeof(float) * 3 * nParticles));
	//gpuErrchk(cudaMemset(dev_vz, 0, sizeof(float) * 3 * nParticles));
	curandState* dev_curand;
	int threads = 32;

	int blocks = (nParticles + threads - 1) / threads;
	gpuErrchk(cudaMalloc((void**)&dev_curand, nParticles * sizeof(curandState)));

	initializeRandomKern << < blocks, threads >> > (nParticles, dev_curand);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_x, dev_curand, -5.f, 5.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_y, dev_curand, 8.f, 258.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_z, dev_curand, -5.f, 5.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vx, dev_curand, -10.f, 10.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vy, dev_curand, -10.f, 10.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	
	fillRandomKern << <blocks, threads >> > (nParticles, dev_vz, dev_curand, -10.f, 10.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


	//TEST FOR 2 PARTICLES
	/*float tmp[2] = { 5.f, -10.f };
	gpuErrchk(cudaMemcpy(dev_x, &tmp[0], sizeof(float) * 2, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_y, &tmp[0], sizeof(float) * 2, cudaMemcpyHostToDevice));

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	tmp[0] = -5.f;
	tmp[1] = 5.f;
	cudaMemcpy(dev_vx, &tmp[0], sizeof(float) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vy, &tmp[0], sizeof(float) * 2, cudaMemcpyHostToDevice);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());*/

}