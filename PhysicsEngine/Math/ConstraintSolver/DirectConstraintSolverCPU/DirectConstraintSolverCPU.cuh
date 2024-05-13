#pragma once
#include "../ConstraintSolver.cuh"


class DirectConstraintSolverCPU : public ConstraintSolver 
{
	float* x_cpu, * y_cpu, * z_cpu, * invmass_cpu;
	float lambda[2000];
public:
	DirectConstraintSolverCPU(int nParticles);
	virtual ~DirectConstraintSolverCPU();

	virtual void calculateForces(
		float* new_x, float* new_y, float* new_z,
		float* invmass, int* dev_phase, float dt, int iterations
	) override;

	virtual void calculateStabilisationForces(
		float* x, float* y, float* z, int* mode,
		float* new_x, float* new_y, float* new_z,
		float* invmass, float dt, int iterations
	) override;

	template<typename T>
	int projectConstraints(float* x, float* y, float* z, float* invmass, int* phase, float dt, int iterations, int lambda_offset);
};
