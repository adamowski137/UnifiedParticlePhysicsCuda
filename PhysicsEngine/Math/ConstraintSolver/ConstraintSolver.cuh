#pragma once

#include "../../Constraint/ConstraintArgs.hpp"

class ConstraintSolver {
public:
	ConstraintSolver(int particles);
	virtual ~ConstraintSolver();
	virtual void calculateForces(float dt, int iterations) = 0;
	virtual void calculateStabilisationForces(float dt, int iterations) = 0;
	

	void initConstraintArgsBuilder(
		float* x, float* y, float* z,
		float* new_x, float* new_y, float* new_z,
		int* SDF_mode, float* SDF_value, float* SDF_normal_x, float* SDF_normal_y, float* SDF_normal_z,
		float* invmass);
	void clearAllConstraints();

protected:
	float* dev_dx;
	float* dev_dy;
	float* dev_dz;
	
	ConstraintArgsBuilder builder;


	int nParticles;
};

