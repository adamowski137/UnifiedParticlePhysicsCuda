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

__host__ __device__ void DistanceConstrain::positionDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz, int index, float* output)
{

	float distX = (x[p1] - x[p2]) * (x[p1] - x[p2]);
	float distY = (y[p1] - y[p2]) * (y[p1] - y[p2]);
	float distZ = (z[p1] - z[p2]) * (z[p1] - z[p2]);

	float l = sqrtf(distX + distY + distZ);

	if (index == 0)
	{
		output[0] = (x[p1] - x[p2]) / l;
		output[1] = (y[p1] - y[p2]) / l;
		output[2] = (z[p1] - z[p2]) / l;
	}
	else
	{
		output[0] = (x[p2] - x[p1]) / l;
		output[1] = (y[p2] - y[p1]) / l;
		output[2] = (z[p2] - z[p1]) / l;
	}
}
