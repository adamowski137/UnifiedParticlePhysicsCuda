#include "DistanceConstraint.cuh"
#include <cmath>
#include "../Constraint.cuh"

__host__ __device__ DistanceConstraint DistanceConstraint::init(float d, int p1, int p2, ConstraintLimitType type, float compliance)
{
	((Constraint*)this)->init(2, compliance, type);
	this->p[0] = p1;
	this->p[1] = p2;
	this->d = d;
	return *this;
}

__host__ __device__ float DistanceConstraint::operator()(float* x, float* y, float* z, float dt)
{
	float distX = (x[p[0]] - x[p[1]]) * (x[p[0]] - x[p[1]]);
	float distY = (y[p[0]] - y[p[1]]) * (y[p[0]] - y[p[1]]);
	float distZ = (z[p[0]] - z[p[1]]) * (z[p[0]] - z[p[1]]);

	float C = (sqrtf(distX + distY + distZ) - d);
	return k * C;
}

__host__ __device__ void DistanceConstraint::positionDerivative(float* x, float* y, float* z, float* jacobian, int nParticles, int index)
{
	float distX = (x[p[0]] - x[p[1]]) * (x[p[0]] - x[p[1]]);
	float distY = (y[p[0]] - y[p[1]]) * (y[p[0]] - y[p[1]]);
	float distZ = (z[p[0]] - z[p[1]]) * (z[p[0]] - z[p[1]]);

	float d = sqrtf(distX + distY + distZ);
	//d = 1;

	int idx1 = index * 3 * nParticles + 3 * p[0];
	int idx2 = index * 3 * nParticles + 3 * p[1];

	float dCx = (x[p[0]] - x[p[1]]) / d;
	float dCy = (y[p[0]] - y[p[1]]) / d;
	float dCz = (z[p[0]] - z[p[1]]) / d;

	jacobian[idx1 + 0] = dCx;
	jacobian[idx1 + 1] = dCy;
	jacobian[idx1 + 2] = dCz;

	jacobian[idx2 + 0] = -dCx;
	jacobian[idx2 + 1] = -dCy;
	jacobian[idx2 + 2] = -dCz;
}
