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

	float len = sqrtf(s.a * s.a + s.b * s.b + s.c * s.c);
	float dist = (s.a * x[p[0]] + s.b * y[p[0]] + s.c * z[p[0]] + s.d) / len;
	return dist - r;
}

__host__ __device__ float SurfaceConstraint::timeDerivative(float* x, float* y, float* z, float* vx, float* vy, float* vz)
{
	float val = s.a * vx[p[0]] + s.b * vy[p[0]] + s.c * vz[p[0]] + s.d;
	float res = s.normal[0] * vx[p[0]] + s.normal[1] * vy[p[0]] + s.normal[2] * vz[p[0]];
	return val < 0 ? -res : res;
}

__host__ __device__ void SurfaceConstraint::positionDerivative(float* x, float* y, float* z, float* vx, float* vy, float* vz, int index, float* output)
{
	output[0] = s.normal[0];
	output[1] = s.normal[1];
	output[2] = s.normal[2];
}

__host__ __device__ void SurfaceConstraint::timePositionDerivative(float* x, float* y, float* z, float* vx, float* vy, float* vz, int index, float* output)
{
	output[0] = 0;
	output[1] = 0;
	output[2] = 0;
}
