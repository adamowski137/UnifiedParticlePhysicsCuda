#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "../Constrain/DistanceConstrain/DistanceConstrain.cuh"
class ConstrainSolver {
public:
	ConstrainSolver(int particles);
	~ConstrainSolver();
	void calculateForces(
		float* x, float* y, float* z,
		float* vx, float* vy, float* vz,
		float* invmass, float* fc, float dt
	);
	void setConstraints(std::vector<std::pair<int, int>> pairs, float d);

private:
	float* dev_jacobian;
	float* dev_jacobian_transposed;
	float* dev_velocity_jacobian;
	float* dev_A;
	float* dev_b;
	float* dev_lambda;
	float* dev_new_lambda;
	int nParticles;
	int nConstraints;
	DistanceConstrain* dev_constrains;
	std::vector<DistanceConstrain> cpu_constraints;
};
