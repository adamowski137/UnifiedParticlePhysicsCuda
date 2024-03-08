#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "../Constrain/DistanceConstrain/DistanceConstrain.cuh"
class ConstrainSolver {
public:
	ConstrainSolver(int particles, int constrainsNumber);
	~ConstrainSolver();
	void calculateForces(
		float* x, float* y, float* z,
		float* new_x, float* new_y, float* new_z,
		float* vx, float* vy, float* vz,
		float* invmass, float* fc, float dt
	);
private:
	float* dev_jacobian;
	float* dev_jacobian_transposed;
	float* dev_velocity_jacobian;
	float* dev_A;
	float* dev_b;
	float* dev_lambda;
	float* dev_new_lambda;
	int particles;
	int constrainsNumber;
	DistanceConstrain* dev_constrains;
};
