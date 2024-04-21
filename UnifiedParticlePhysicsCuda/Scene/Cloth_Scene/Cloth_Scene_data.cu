#include "Cloth_Scene_data.cuh"
#include "../../PhysicsEngine/Particle/ParticleData.cuh"
#include "../../PhysicsEngine/GpuErrorHandling.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include "../../PhysicsEngine/Cloth/Cloth.hpp"
#include <vector>

void initData_ClothScene(int nParticles,
	float* dev_x, float* dev_y, float* dev_z,
	float* dev_vx, float* dev_vy, float* dev_vz, int* mode)
{
	curandState* dev_curand;
	int threads = 32;

	int blocks = (nParticles + threads - 1) / threads;
	gpuErrchk(cudaMalloc((void**)&dev_curand, nParticles * sizeof(curandState)));

	initializeRandomKern << < blocks, threads >> > (nParticles, dev_curand);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	float d = 3;
	int W = 10;
	int H = 12;
	Cloth::initClothSimulation(H, W, d, -d * W / 2.f, 0.f, 0.f, dev_x, dev_y, dev_z);

	std::vector<int> modes(nParticles, 0);
	//for (int i = 0; i < W; i++)
	//	modes[i] = 1;
	modes[0] = 1;
	modes[W - 1] = 1;
	gpuErrchk(cudaMemcpy(mode, modes.data(), modes.size() * sizeof(int), cudaMemcpyHostToDevice));




	
	fillRandomKern << <blocks, threads >> > (nParticles, dev_z, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vx, dev_curand, -0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	fillRandomKern << <blocks, threads >> > (nParticles, dev_vy, dev_curand, 0.f, 0.f);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

}