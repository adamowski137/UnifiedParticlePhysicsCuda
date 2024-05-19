#pragma once
#include "../Constraint.cuh"
#include "../ConstraintArgs.hpp"

class RigidBodyConstraint : public Constraint
{

public:
	int* p;
	RigidBodyConstraint(float* x, float* y, float* z, float* m, int* p, int n, ConstraintLimitType type, float compliance = 0.f);
	~RigidBodyConstraint();

	bool calculateShapeCovariance(float* x, float* y, float* z, float* invmass);
	void calculatePositionChange(ConstraintArgs args);
private:
	float* rx;
	float* ry;
	float* rz;
	float* tmp;
	float* decompostion;
	float totalMass;
	float cx, cy, cz;
	float* tcx;
	float* tcy;
	float* tcz;
};