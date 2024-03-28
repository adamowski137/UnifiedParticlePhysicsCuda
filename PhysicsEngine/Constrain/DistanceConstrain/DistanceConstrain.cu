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


__host__ __device__ float DistanceConstrain::timeDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz)
{
	float distX = (x[p[0]] - x[p[1]]) * (x[p[0]] - x[p[1]]);
	float distY = (y[p[0]] - y[p[1]]) * (y[p[0]] - y[p[1]]);
	float distZ = (z[p[0]] - z[p[1]]) * (z[p[0]] - z[p[1]]);

	float nx = (x[p[0]] - x[p[1]]);
	float ny = (y[p[0]] - y[p[1]]);
	float nz = (z[p[0]] - z[p[1]]);

	float len = sqrt(nx * nx + ny * ny + nz * nz);

	nx /= len;
	ny /= len;
	nz /= len;

	float diffvX = (vx[p[0]] - vx[p[1]]);
	float diffvY = (vy[p[0]] - vy[p[1]]);
	float diffvZ = (vz[p[0]] - vz[p[1]]);

	//return diffvX + diffvY + diffvZ;

	//return sqrt(diffvX * diffvX + diffvY * diffvY + diffvZ * diffvZ);
	float coeff = 1 / sqrtf(distX + distY + distZ);
	return coeff * (nx * vx[p[0]] + ny * vy[p[0]] + nz * vz[p[0]] - nx * vx[p[1]] - ny * vy[p[1]] - nz * vz[p[1]]);
}


__host__ __device__ void DistanceConstrain::positionDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz, int index, float* output)
{
	if (index == 0)
	{
		output[0] = x[p[0]] - x[p[1]];
		output[1] = y[p[0]] - y[p[1]];
		output[2] = z[p[0]] - z[p[1]];
	}
	else
	{
		output[0] = x[p[1]] - x[p[0]];
		output[1] = y[p[1]] - y[p[0]];
		output[2] = z[p[1]] - z[p[0]];
	}
}


__host__ __device__ void DistanceConstrain::timePositionDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz, int index, float* output)
{
	output[0] = 0.0f;
	output[1] = 0.0f;
	output[2] = 0.0f;
}





