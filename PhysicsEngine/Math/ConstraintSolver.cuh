#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <thrust/device_ptr.h>
#include "../Constraint/ConstraintStorage.cuh"
#include "../Constraint/DistanceConstraint/DistanceConstraint.cuh"
#include "../Constraint/SurfaceConstraint/SurfaceConstraint.cuh"
#include "../List/List.cuh"
#include "LinearSolver.cuh"

template<typename T>
void fillJacobiansWrapper(int nConstraints, int nParticles,
	float* x, float* y, float* z,
	float* dx, float* dy, float* dz,
	float* jacobian,
	T* constrains, ConstraintType type, int iterations);


class ConstraintSolver {
public:
	ConstraintSolver(int particles);
	~ConstraintSolver();
	void calculateForces(
		float* new_x, float* new_y, float* new_z,
		float* invmass, float dt, int iterations
	);
	void calculateStabilisationForces(
		float* x, float* y, float* z,
		float* new_x, float* new_y, float* new_z,
		float* invmass, float dt, int iterations
	);

	void setStaticConstraints(std::vector<std::pair<int, int>> pairs, float d);
	void addDynamicConstraints(List* collisions, int* counts, float d, ConstraintLimitType type);
	void addSurfaceConstraints(SurfaceConstraint* surfaceConstraints, int nSurfaceConstraints);

private:
	float* dev_dx;
	float* dev_dy;
	float* dev_dz;

	// J matrix, dynamically created in every iteration
	float* dev_jacobian;
	float* dev_jacobian_transposed;

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

	void allocateArrays(int size);
	
	template<typename T>
	void projectConstraints(float* invmass, float* x, float* y, float* z, float dt, ConstraintType type, bool dynamic, int iterations);
	void clearArrays(int nConstraints);
};




template<typename T>
void ConstraintSolver::projectConstraints(float* invmass, float* x, float* y, float* z, float dt, ConstraintType type, bool dynamic, int iterations)
{
	std::pair<T*, int> constraints = ConstraintStorage::Instance.getConstraints<T>(type, dynamic);
	int nConstraints = constraints.second;
	if (nConstraints == 0) return;
	this->allocateArrays(nConstraints);
	
	fillJacobiansWrapper<T>(
		nConstraints, nParticles, 
		x, y, z,
		dev_dx, dev_dy, dev_dz,
		dev_jacobian,
		dev_jacobian_transposed, dev_A, dev_b, dt,
		invmass,  dev_lambda, dev_new_lambda,
		dev_c_min, dev_c_max,
		constraints.first, type, iterations);

	clearArrays(nConstraints);
}

