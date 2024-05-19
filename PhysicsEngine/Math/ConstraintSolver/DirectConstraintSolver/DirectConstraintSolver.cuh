#pragma once
#include "../ConstraintSolver.cuh"
#include "../../../Constraint/ConstraintArgs.hpp"


class DirectConstraintSolver : public ConstraintSolver 
{
	int* dev_nConstraintsPerParticle;
	float* dev_delta_lambda;
	int nConstraintsMaxAllocated;
public:
	DirectConstraintSolver(int nParticles);
	virtual ~DirectConstraintSolver();

	virtual void calculateForces(
		float* x, float* y, float* z, int* mode,
		float* new_x, float* new_y, float* new_z,
		float* invmass, float dt, int iterations
	) override;

	virtual void calculateStabilisationForces(
		float* x, float* y, float* z, int* mode,
		float* new_x, float* new_y, float* new_z,
		float* invmass, float dt, int iterations
	) override;

	template<typename T>
	void projectConstraints(ConstraintArgs args);

	void applyOffset(float* x, float* y, float* z);
};
