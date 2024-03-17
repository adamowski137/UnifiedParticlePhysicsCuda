#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <thrust/device_ptr.h>
#include "../Constrain/DistanceConstrain/DistanceConstrain.cuh"
#include "../Constrain/SurfaceConstraint/SurfaceConstraint.cuh"
#include "../List/List.cuh"

class ConstrainSolver {
public:
	ConstrainSolver(int particles);
	~ConstrainSolver();
	void calculateForces(
		float* x, float* y, float* z,
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
	int nConstraints;
	int nStaticConstraints;
	int nDynamicConstraints;
	int nSurfaceConstraints;

	// if number of constraints decreased between simulation steps we do not want to reallocate arrays
	int nConstraintsMaxAllocated;

	// mainly collision constraints
	DistanceConstrain* dev_constraints;
	DistanceConstrain* dev_staticConstraints;
	DistanceConstrain* dev_dynamicConstraints;

	SurfaceConstraint* dev_surfaceConstraints;

	void allocateArrays();
	void projectConstraints(float* x, float* y, float* z, float* vx, float* vy, float* vz, float dt);
};
