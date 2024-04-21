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
	return C;
}

__host__ __device__ void DistanceConstraint::positionDerivative(float* x, float* y, float* z, int index, float* output)
{
	float distX = (x[p[0]] - x[p[1]]) * (x[p[0]] - x[p[1]]);
	float distY = (y[p[0]] - y[p[1]]) * (y[p[0]] - y[p[1]]);
	float distZ = (z[p[0]] - z[p[1]]) * (z[p[0]] - z[p[1]]);

	float d = sqrtf(distX + distY + distZ);
	//d = 1;

	if (index == 0)
	{
		output[0] = (x[p[0]] - x[p[1]]) / d;
		output[1] = (y[p[0]] - y[p[1]]) / d;
		output[2] = (z[p[0]] - z[p[1]]) / d;
	}
	else
	{
		output[0] = (x[p[1]] - x[p[0]]) / d;
		output[1] = (y[p[1]] - y[p[0]]) / d;
		output[2] = (z[p[1]] - z[p[0]]) / d;
	}
}
