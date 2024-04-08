#include "SurfaceConstraint.cuh"
#include <cmath>

__host__ __device__ SurfaceConstraint SurfaceConstraint::init(float d, int particle, Surface s)
{
	((Constrain*)this)->init(1, 1.0f, ConstraintLimitType::GEQ);
	this->r = d;
	this->p[0] = particle;
	this->s = s;
	return *this;
}

__host__ __device__ float SurfaceConstraint::operator()(float* x, float* y, float* z, float* vx, float* vy, float* vz)
{
	//return (s.normal[0] * x[p] + s.normal[1] * y[p] + s.normal[2] * z[p]) - r;
	return (x[p[0]] * s.a + y[p[0]] * s.b + z[p[0]] * s.c + s.d) / s.abc_root - r;

}

__host__ __device__ void SurfaceConstraint::positionDerivative(float* x, float* y, float* z, float* vx, float* vy, float* vz, int index, float* output)
{
	output[0] = s.normal[0];
	output[1] = s.normal[1];
	output[2] = s.normal[2];
}