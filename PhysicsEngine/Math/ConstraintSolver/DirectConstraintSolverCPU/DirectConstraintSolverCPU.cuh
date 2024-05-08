#pragma once
#include "../ConstraintSolver.cuh"


class DirectConstraintSolverCPU : public ConstraintSolver 
{
public:
	DirectConstraintSolver(int nParticles);
	virtual ~DirectConstraintSolver();

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
	void projectConstraints(float* x, float* y, float* z, float* invmass, int* phase, float dt, int iterations);

	void applyOffset(float* x, float* y, float* z);
};
