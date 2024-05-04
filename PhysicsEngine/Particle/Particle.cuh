#pragma once
#define THREADS 512
#include <memory>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include "../Constraint/Constraint.cuh"
#include "../Collision/CollisionGrid.cuh"
#include "../Collision/SurfaceCollisionFinder.cuh"
#include "../Math/ConstraintSolver.cuh"

#define ANY_CONSTRAINTS_ON 1
#define GRID_CHECKING_ON 2
#define SURFACE_CHECKING_ON 4



class ParticleType
{
public:
	ParticleType(int amount, int mode);
	~ParticleType();

	void mapCudaVBO(unsigned int vbo);
	void renderData(unsigned int vbo);
	void calculateNewPositions(float dt);
	void setConstraints(std::vector<std::pair<int, int>> pairs, float d);
	void setSurfaces(std::vector<Surface> surfaces);
	void setExternalForces(float fx, float fy, float fz);
	void setRigidBodyConstraint(std::vector<int> points);
	inline int particleCount() { return nParticles; }

private:
	const int nParticles;
	int mode;
	int* dev_phase;
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
	friend class Scene;
	curandState* dev_curand;
	
	float fextx, fexty, fextz;

	std::unique_ptr<ConstraintSolver> constraintSolver;
	std::unique_ptr<CollisionGrid> collisionGrid;
	std::unique_ptr<SurfaceCollisionFinder> surfaceCollisionFinder;

	int blocks;

	void setupDeviceData();
	void allocateDeviceData();
};