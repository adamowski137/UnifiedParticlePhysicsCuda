#pragma once
#define THREADS 512
#include <curand.h>
#include <curand_kernel.h>
#include <memory>
#include "../Constrain/Constrain.cuh"
#include <vector>

class ParticleType
{
public:
	ParticleType(int amount);
	~ParticleType();


	void mapCudaVBO(unsigned int vbo);
	void renderData(unsigned int vbo);
	void calculateNewPositions(float dt);
private:
	const int amountOfParticles;
	float* dev_x;
	float* dev_y;
	float* dev_z;
	float* dev_new_x;
	float* dev_new_y;
	float* dev_new_z;
	float* dev_vx;
	float* dev_vy;
	float* dev_vz;
	float* dev_invmass;

	float* dev_jacobian;

	curandState* dev_curand;
	
	float* dev_invM;

	float fextx, fexty, fextz;

	std::vector<std::shared_ptr<Constrain>> constrains;

	int blocks;

	void setupDeviceData();
};