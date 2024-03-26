#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <thrust/device_ptr.h>
#include "../Constrain/ConstrainStorage.cuh"
#include "../Constrain/DistanceConstrain/DistanceConstrain.cuh"
#include "../Constrain/SurfaceConstraint/SurfaceConstraint.cuh"
#include "../List/List.cuh"
#include "LinearSolver.cuh"

template<typename T>
void fillJacobiansWrapper(int nConstraints, int nParticles,
	float* x, float* y, float* z,
	float* vx, float* vy, float* vz,
	float* jacobian, float* velocity_jacobian,
	T* constrains, ConstrainType type);


class ConstrainSolver {
public:
	ConstrainSolver(int particles);
	~ConstrainSolver();
	void calculateForces(
		float* x, float* y, float* z,
		float* new_x, float* new_y, float* new_z,
		float* vx, float* vy, float* vz,
		float* invmass, float* fc, float dt
	);

	void setStaticConstraints(std::vector<std::pair<int, int>> pairs, float d);
	void addDynamicConstraints(List* collisions, int* counts, float d, ConstraintLimitType type);
	void addSurfaceConstraints(SurfaceConstraint* surfaceConstraints, int nSurfaceConstraints);

private:
	// J matrix, dynamically created in every iteration
	float* dev_jacobian;
	float* dev_jacobian_transposed;

	// J dot matrix
	float* dev_velocity_jacobian;

	// as in Ax = b matrix equation
	float* dev_A;
	float* dev_b;

	float* dev_c_min;
	float* dev_c_max;

	// coefficients, results of matrix equation, multiplied by J give force that has to be applied to particles
	float* dev_lambda;
	float* dev_new_lambda;

	int nParticles;

	// if number of constraints decreased between simulation steps we do not want to reallocate arrays
	int nConstraintsMaxAllocated;

	// mainly collision constraints

	SurfaceConstraint* dev_surfaceConstraints;

	void allocateArrays(int size);
	
	template<typename T>
	void projectConstraints(float* fc, float* invmass, float* x, float* y, float* z, float* vx, float* vy, float* vz, float dt, ConstrainType type);

};




template<typename T>
void ConstrainSolver::projectConstraints(float* fc, float* invmass, float* x, float* y, float* z, float* vx, float* vy, float* vz, float dt, ConstrainType type)
{
	std::pair<T*, int> constraints = ConstrainStorage::Instance.getConstraints<T>(type);
	int nConstraints = constraints.second;
	if (nConstraints == 0) return;
	this->allocateArrays(nConstraints);

	fillJacobiansWrapper(
		nConstraints, nParticles, 
		x, y, z, vx, vy, vz, 
		dev_jacobian, dev_velocity_jacobian,
		dev_jacobian_transposed, dev_A, dev_b, dt,
		invmass, fc, dev_lambda, dev_new_lambda,
		dev_c_min, dev_c_max,
		constraints.first, type);

}
