#pragma once
#define THREADS 512
#include <curand.h>
#include <curand_kernel.h>
#include <memory>
#include "../Constrain/Constrain.cuh"
#include <vector>
#include "../Math/ConstrainSolver.cuh"
#include "../Collision/CollisionGrid.cuh"
#include "../Collision/SurfaceCollisionFinder.cuh"

class ParticleType
{
public:
	ParticleType(int amount);
	~ParticleType();


	void mapCudaVBO(unsigned int vbo);
	void renderData(unsigned int vbo);
	void calculateNewPositions(float dt);
	void setConstraints(std::vector<std::pair<int, int>> pairs, float d);
	inline int particleCount() { return nParticles; }
private:
	const int nParticles;
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
	float* dev_fc;
	
	List* dev_collisions;
	int* dev_sums;
	curandState* dev_curand;
	
	float fextx, fexty, fextz;

	std::unique_ptr<ConstrainSolver> constrainSolver;
	std::unique_ptr<CollisionGrid> collisionGrid;
	std::unique_ptr<SurfaceCollisionFinder> surfaceCollisionFinder;

	int blocks;

	void setupDeviceData();
};