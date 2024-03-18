#include "DistanceConstrain.cuh"
#include <cmath>
#include "../Constrain.cuh"

__host__ __device__ DistanceConstrain::DistanceConstrain(float d, int p1, int p2, ConstraintLimitType type) : Constrain{ 2, 1.0f, type }, d{ d }, p1{ p1 }, p2 { p2 }
{
}


__host__ __device__ float DistanceConstrain::operator()(float* x, float* y, float* z,
	float* vx, float* vy, float* vz)
{
	float distX = (x[p1] - x[p2]) * (x[p1] - x[p2]);
	float distY = (y[p1] - y[p2]) * (y[p1] - y[p2]);
	float distZ = (z[p1] - z[p2]) * (z[p1] - z[p2]);

	return sqrtf(distX + distY + distZ) - d;
}


__host__ __device__ float DistanceConstrain::timeDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz)
{
	float distX = (x[p1] - x[p2]) * (x[p1] - x[p2]);
	float distY = (y[p1] - y[p2]) * (y[p1] - y[p2]);
	float distZ = (z[p1] - z[p2]) * (z[p1] - z[p2]);

	float nx = (x[p1] - x[p2]);
	float ny = (y[p1] - y[p2]);
	float nz = (z[p1] - z[p2]);

	float len = sqrt(nx * nx + ny * ny + nz * nz);

	nx /= len;
	ny /= len;
	nz /= len;

	float diffvX = (vx[p1] - vx[p2]);
	float diffvY = (vy[p1] - vy[p2]);
	float diffvZ = (vz[p1] - vz[p2]);

	//return diffvX + diffvY + diffvZ;

	//return sqrt(diffvX * diffvX + diffvY * diffvY + diffvZ * diffvZ);
	float coeff = 1 / sqrtf(distX + distY + distZ);
	return coeff * (nx * vx[p1] + ny * vy[p1] + nz * vz[p1] - nx * vx[p2] - ny * vy[p2] - nz * vz[p2]);
}


__host__ __device__ void DistanceConstrain::positionDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz, int index, float* output)
{
	if (index == 0)
	{
		output[0] = x[p1] - x[p2];
		output[1] = y[p1] - y[p2];
		output[2] = z[p1] - z[p2];
	}
	else
	{
		output[0] = x[p2] - x[p1];
		output[1] = y[p2] - y[p1];
		output[2] = z[p2] - z[p1];
	}
}


__host__ __device__ void DistanceConstrain::timePositionDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz, int index, float* output)
{
	output[0] = 0.0f;
	output[1] = 0.0f;
	output[2] = 0.0f;
}





