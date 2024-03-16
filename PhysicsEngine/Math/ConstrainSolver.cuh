#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <thrust/device_ptr.h>
#include "../Constrain/DistanceConstrain/DistanceConstrain.cuh"
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
	void addDynamicConstraint(int idx1, int idx2, float d, ConstraintLimitType type);

private:
	// J matrix, dynamically created in every iteration
	float* dev_jacobian;
	float* dev_jacobian_transposed;

	// J dot matrix
	float* dev_velocity_jacobian;

	// as in Ax = b matrix equation
	float* dev_A;
	float* dev_b;

	// coefficients, results of matrix equation, multiplied by J give force that has to be applied to particles
	float* dev_lambda;
	float* dev_new_lambda;

	int nParticles;
	int nConstraints;
	int nStaticConstraints;
	int nDynamicConstraints;

	// if number of constraints decreased between simulation steps we do not want to reallocate arrays
	int nConstraintsMaxAllocated;

	// mainly collision constraints
	DistanceConstrain* dev_constraints;
	DistanceConstrain* dev_staticConstraints;
	std::vector<DistanceConstrain> dynamicConstraints;

	void allocateArrays();
	void projectConstraints(float* x, float* y, float* z, float* vx, float* vy, float* vz);
};
