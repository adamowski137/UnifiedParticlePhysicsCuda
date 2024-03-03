#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "../Constrain/Constrain.cuh"

class ConstrainSolver {
public:
	ConstrainSolver(int particles, int constrainsNumber, Constrain** constrains);
	~ConstrainSolver();
private:
	std::vector<Constrain> constrains;
	float* dev_jacobian;
	float* dev_velocity_jacobian;
	int particles;
	int constrainsNumber;
	Constrain** dev_constrains;

	void fillJacobians(
		float* x, float* y, float* z,
		float* vx, float* vy, float* vz
	);
};
