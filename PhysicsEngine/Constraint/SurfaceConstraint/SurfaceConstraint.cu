#include "SurfaceConstraint.cuh"
#include <cmath>

__host__ __device__ SurfaceConstraint SurfaceConstraint::init(float d, int particle, Surface s)
{
	((Constraint*)this)->init(1, 5.0f, ConstraintLimitType::GEQ);
	this->r = d;
	this->p[0] = particle;
	this->s = s;
	return *this;
}

__host__ __device__ float SurfaceConstraint::operator()(float* x, float* y, float* z)
{
	float C = x[p[0]] * s.a + y[p[0]] * s.b + z[p[0]] * s.c + s.d / s.abc_root - r;
	return k * C;
}

__host__ __device__ void SurfaceConstraint::positionDerivative(float* x, float* y, float* z, float* jacobian, int nParticles, int index)
{
	int idx = index * 3 * nParticles + 3 * p[0];

	jacobian[idx + 0] = s.normal[0];
	jacobian[idx + 1] = s.normal[1];
	jacobian[idx + 2] = s.normal[2];
}

__device__ void SurfaceConstraint::directSolve(float* x, float* y, float* z, float* dx, float* dy, float* dz, float* invmass, int* nConstraintsPerParticle)
{
	float C = (*this)(x, y, z) / k * 0.5f;

	atomicAdd(dx + p[0], -C * s.normal[0]);
	atomicAdd(dy + p[0], -C * s.normal[1]);
	atomicAdd(dz + p[0], -C * s.normal[2]);

	atomicAdd(nConstraintsPerParticle + p[0], 1);
	//dx[p[0]] += -C * s.normal[0];
	//dy[p[0]] += -C * s.normal[1];
	//dz[p[0]] += -C * s.normal[2];
}