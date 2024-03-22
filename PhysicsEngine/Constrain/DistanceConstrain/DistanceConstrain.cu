#include "DistanceConstrain.cuh"
#include <cmath>
#include "../Constrain.cuh"

__host__ __device__ DistanceConstrain DistanceConstrain::init(float d, int p1, int p2, ConstraintLimitType type)
{
	((Constrain*)this)->init(2, 1.0f, type);
	this->p[0] = p1;
	this->p[1] = p2;
	this->d = d;
	return *this;
}

__host__ __device__ float DistanceConstrain::operator()(float* x, float* y, float* z,
	float* vx, float* vy, float* vz)
{
	float distX = (x[p[0]] - x[p[1]]) * (x[p[0]] - x[p[1]]);
	float distY = (y[p[0]] - y[p[1]]) * (y[p[0]] - y[p[1]]);
	float distZ = (z[p[0]] - z[p[1]]) * (z[p[0]] - z[p[1]]);

	return sqrtf(distX + distY + distZ) - d;
}

__host__ __device__ void DistanceConstrain::positionDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz, int index, float* output)
{

	float distX = (x[p[0]] - x[p[1]]) * (x[p[0]] - x[p[1]]);
	float distY = (y[p[0]] - y[p[1]]) * (y[p[0]] - y[p[1]]);
	float distZ = (z[p[0]] - z[p[1]]) * (z[p[0]] - z[p[1]]);

	float l = sqrtf(distX + distY + distZ);

	if (index == 0)
	{
		output[0] = (x[p[0]] - x[p[1]]) / l;
		output[1] = (y[p[0]] - y[p[1]]) / l;
		output[2] = (z[p[0]] - z[p[1]]) / l;
	}
	else
	{
		output[0] = (x[p[1]] - x[p[0]]) / l;
		output[1] = (y[p[1]] - y[p[0]]) / l;
		output[2] = (z[p[1]] - z[p[0]]) / l;
	}
}
