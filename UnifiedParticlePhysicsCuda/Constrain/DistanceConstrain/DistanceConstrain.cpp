#include "DistanceConstrain.hpp"

DistanceConstrain::DistanceConstrain(float k, float d) : Constrain(2, k, true), d{d}
{

}

float DistanceConstrain::operator()(int* index, float* x, float* y, float* z)
{
	int distx = x[index[0]] - x[index[1]];
	int disty = y[index[0]] - y[index[1]];
	int distz = z[index[0]] - z[index[1]];
	return (distx * distx + disty * disty + distz * distz) - d;
}
