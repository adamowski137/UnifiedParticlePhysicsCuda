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

	virtual void calculateForces(float dt, int iterations) override;
	virtual void calculateStabilisationForces(float dt, int iterations) override;

	template<typename T>
	void projectConstraints(ConstraintArgs args);

	void applyOffset(float* x, float* y, float* z);
};
