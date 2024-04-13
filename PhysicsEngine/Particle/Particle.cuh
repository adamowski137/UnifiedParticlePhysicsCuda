#pragma once
#define THREADS 512
#include <memory>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include "../Constraint/Constraint.cuh"
#include "../Math/ConstraintSolver.cuh"


class ParticleType
{
public:
	ParticleType(int amount, int mode, void(*setDataFunction)(int, float*, float*, float*, float*, float*, float*) );
	~ParticleType();


	void mapCudaVBO(unsigned int vbo);
	void renderData(unsigned int vbo);
	void calculateNewPositions(float dt);
	void setConstraints(std::vector<std::pair<int, int>> pairs, float d);
	void setSurfaces(std::vector<Surface> surfaces);
	void setExternalForces(float fx, float fy, float fz);
	inline int particleCount() { return nParticles; }
private:
	const int nParticles;
	int mode;
	float* dev_x;
	float* dev_y;
	float* dev_z;
	float* dev_dx;
	float* dev_dy;
	float* dev_dz;
	float* dev_new_x;
	float* dev_new_y;
	float* dev_new_z;
	float* dev_vx;
	float* dev_vy;
	float* dev_vz;
	float* dev_invmass;

	curandState* dev_curand;
	
	float fextx, fexty, fextz;

	std::unique_ptr<ConstraintSolver> constraintSolver;

	int blocks;

	void setupDeviceData(void(*setDataFunction)(int, float*, float*, float*, float*, float*, float*));
	void allocateDeviceData();
};