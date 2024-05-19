#include "SurfaceConstraint.cuh"
#include "../../Config/Config.hpp"
#include "../../Constants.hpp"
#include <cmath>

__host__ __device__ SurfaceConstraint SurfaceConstraint::init(float d, float k, int particle, Surface s)
{
	((Constraint*)this)->init(1, k, ConstraintLimitType::GEQ);
	this->r = d;
	this->p[0] = particle;
	this->s = s;
	return *this;
}

__host__ __device__ float SurfaceConstraint::operator()(float* x, float* y, float* z)
{
	float C = x[p[0]] * s.a + y[p[0]] * s.b + z[p[0]] * s.c + s.d / s.abc_root - r;
	return C;
}

__host__ __device__ void SurfaceConstraint::positionDerivative(float* x, float* y, float* z, float* jacobian, int nParticles, int index)
{
	int idx = index * 3 * nParticles + 3 * p[0];

	jacobian[idx + 0] = s.normal[0];
	jacobian[idx + 1] = s.normal[1];
	jacobian[idx + 2] = s.normal[2];
}

__device__ void SurfaceConstraint::directSolve(float* x, float* y, float* z, float* dx, float* dy, float* dz, float* invmass, int* nConstraintsPerParticle, float dt)
{
	float C = (*this)(x, y, z);

	float lambda = -C * k;
	lambda = min(max(lambda, cMin), cMax);

	atomicAdd(dx + p[0], lambda * s.normal[0]);
	atomicAdd(dy + p[0], lambda * s.normal[1]);
	atomicAdd(dz + p[0], lambda * s.normal[2]);

	atomicAdd(nConstraintsPerParticle + p[0], 1);
	//dx[p[0]] += -C * s.normal[0];
	//dy[p[0]] += -C * s.normal[1];
	//dz[p[0]] += -C * s.normal[2];
}

__host__ void SurfaceConstraint::directSolve_cpu(float* x, float* y, float* z, float* invmass)
{
	float C = (*this)(x, y, z);

	float lambda = -C * k;
	
	lambda = std::fmin(std::fmax(lambda, cMin), cMax);
	
	x[p[0]] += lambda * s.normal[0];
	y[p[0]] += lambda * s.normal[1];
	z[p[0]] += lambda * s.normal[2];
}
