#include "DistanceConstrain.cuh"
#include <cmath>
#include "../Constrain.cuh"

DistanceConstrain::DistanceConstrain(float d, int* indexes) : Constrain{ 2, 1.0f, -10000.0, 10000.0, indexes}, d{d}
{
}

float DistanceConstrain::operator()(float* x, float* y, float* z,
	float* vx, float* vy, float* vz)
{
	int p1 = dev_indexes[0];
	int p2 = dev_indexes[1];
	float distX = (x[p1] - x[p2]) * (x[p1] * x[p2]);
	float distY = (y[p1] - y[p2]) * (y[p1] * y[p2]);
	float distZ = (z[p1] - z[p2]) * (z[p1] * z[p2]);

	return sqrtf(distX + distY + distZ) - d;
}

float DistanceConstrain::timeDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz)
{
	int p1 = dev_indexes[0];
	int p2 = dev_indexes[1];
	float distX = (vx[p1] - vx[p2]) * (vx[p1] * vx[p2]);
	float distY = (vy[p1] - vy[p2]) * (vy[p1] * vy[p2]);
	float distZ = (vz[p1] - vz[p2]) * (vz[p1] * vz[p2]);

	return sqrtf(distX + distY + distZ);
}

float DistanceConstrain::positionDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz, int index)
{
	return 1.0f;
}

float DistanceConstrain::timePositionDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz, int index)
{
	return 0.0f;
}




