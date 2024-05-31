#pragma once
#include "../ConstraintSolver.cuh"
#include "../../../Constraint/ConstraintArgs.hpp"

class DirectConstraintSolverGaussSeidel : public ConstraintSolver
{
	int nConstraintsMaxAllocated;
	int* dev_coloring;
public:
	DirectConstraintSolverGaussSeidel(int nParticles);
	virtual ~DirectConstraintSolverGaussSeidel();

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
private:

	template<typename T>
	void projectConstraints(ConstraintArgs args);

	template<typename T>
	int findColoring(std::pair<T*, int> constraintData);

	template<typename T>
	int getParticlePerConstraint();

	void reallocateColoring(int n);
};

