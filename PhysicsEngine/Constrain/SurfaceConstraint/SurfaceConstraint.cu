#include "SurfaceConstraint.cuh"
#include <cmath>

__host__ __device__ SurfaceConstraint::SurfaceConstraint(float d, int particle, Surface s)  
	: Constrain{ 2, 1.0f, ConstraintLimitType::GEQ }, r{ d }, p{ particle }, s{s}
{
}

__host__ __device__ float SurfaceConstraint::operator()(float* x, float* y, float* z, float* vx, float* vy, float* vz)
{
	//return (s.normal[0] * x[p] + s.normal[1] * y[p] + s.normal[2] * z[p]) - r;
	return abs(x[p] * s.a + y[p] * s.b + z[p] * s.c + s.d) / s.abc_root - r;

}

__host__ __device__ void SurfaceConstraint::positionDerivative(float* x, float* y, float* z, float* vx, float* vy, float* vz, float* output)
{
	output[0] = s.normal[0];
	output[1] = s.normal[1];
	output[2] = s.normal[2];
}