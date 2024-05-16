#pragma once

class ConstraintSolver {
public:
	ConstraintSolver(int particles);
	virtual ~ConstraintSolver();
	virtual void calculateForces(
		float* new_x, float* new_y, float* new_z,
		float* invmass, int* dev_phase, float dt, int iterations
	) = 0;
	virtual void calculateStabilisationForces(
		float* x, float* y, float* z, int* mode,
		float* new_x, float* new_y, float* new_z,
		float* invmass, float dt, int iterations
	) = 0;

	void clearAllConstraints();

protected:
	float* dev_dx;
	float* dev_dy;
	float* dev_dz;

	int nParticles;
};

