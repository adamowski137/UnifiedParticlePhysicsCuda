#include "DistanceConstraint.cuh"
#include <cmath>
#include "../Constraint.cuh"

__host__ __device__ DistanceConstraint DistanceConstraint::init(float d, int p1, int p2, ConstraintLimitType type)
{
	((Constraint*)this)->init(2, 1.0f, type);
	this->p[0] = p1;
	this->p[1] = p2;
	this->d = d;
	return *this;
}

__host__ __device__ float DistanceConstraint::operator()(float* x, float* y, float* z)
{
	float distX = (x[p[0]] - x[p[1]]) * (x[p[0]] - x[p[1]]);
	float distY = (y[p[0]] - y[p[1]]) * (y[p[0]] - y[p[1]]);
	float distZ = (z[p[0]] - z[p[1]]) * (z[p[0]] - z[p[1]]);

	return (sqrtf(distX + distY + distZ) - d);
}

__host__ __device__ void DistanceConstraint::positionDerivative(float* x, float* y, float* z, int index, float* output)
{
	if (index == 0)
	{
		output[0] = (x[p[0]] - x[p[1]]);
		output[1] = (y[p[0]] - y[p[1]]);
		output[2] = (z[p[0]] - z[p[1]]);
	}
	else
	{
		output[0] = (x[p[1]] - x[p[0]]);
		output[1] = (y[p[1]] - y[p[0]]);
		output[2] = (z[p[1]] - z[p[0]]);
	}
}
