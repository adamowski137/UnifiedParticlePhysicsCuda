#include "DistanceConstrain.cuh"
#include <cmath>
#include "../Constrain.cuh"

DistanceConstrain::DistanceConstrain(float d, int* indexes) : Constrain{ 2, 1.0f, -10000.0, 10000.0, indexes }, d{ d }
{
}

float DistanceConstrain::operator()(float* x, float* y, float* z,
	float* vx, float* vy, float* vz)
{
	int p1 = dev_indexes[0];
	int p2 = dev_indexes[1];
	float distX = (x[p1] - x[p2]) * (x[p1] - x[p2]);
	float distY = (y[p1] - y[p2]) * (y[p1] - y[p2]);
	float distZ = (z[p1] - z[p2]) * (z[p1] - z[p2]);

	return sqrtf(distX + distY + distZ) - d;
}

float DistanceConstrain::timeDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz)
{
	int p1 = dev_indexes[0];
	int p2 = dev_indexes[1];
	float distX = (x[p1] - x[p2]) * (x[p1] - x[p2]);
	float distY = (y[p1] - y[p2]) * (y[p1] - y[p2]);
	float distZ = (z[p1] - z[p2]) * (z[p1] - z[p2]);

	float diffvX = (vx[p1] - vx[p2]);
	float diffvY = (vy[p1] - vy[p2]);
	float diffvZ = (vz[p1] - vz[p2]);

	//return sqrt(diffvX * diffvX + diffvY * diffvY + diffvZ * diffvZ);
	return 1 / sqrtf(distX + distY + distZ) * (diffvX + diffvY + diffvZ);
}

void DistanceConstrain::positionDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz, int index, float* output)
{
	int p1 = dev_indexes[0];
	int p2 = dev_indexes[1];
	//float len = sqrtf((x[p1] - x[p2]) * (x[p1] - x[p2]) + (y[p1] - y[p2]) * (y[p1] - y[p2]) + (z[p1] - z[p2]) * (z[p1] - z[p2]));
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

void DistanceConstrain::timePositionDerivative(float* x, float* y, float* z,
	float* vx, float* vy, float* vz, int index, float* output)
{
	output[0] = 0.0f;
	output[1] = 0.0f;
	output[2] = 0.0f;
}




