#include "SurfaceConstraint.cuh"
#include "../../Config/Config.hpp"
#include "../../Constants.hpp"
#include <cmath>
#include <stdio.h>

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

__device__ void SurfaceConstraint::directSolve(ConstraintArgs args)
{
	float C = (*this)(args.x, args.y, args.z);

	float lambda = -C * k;
	lambda = min(max(lambda, cMin), cMax);

	atomicAdd(args.dx + p[0], lambda * s.normal[0]);
	atomicAdd(args.dy + p[0], lambda * s.normal[1]);
	atomicAdd(args.dz + p[0], lambda * s.normal[2]);

	atomicAdd(args.nConstraintsPerParticle + p[0], 1);

	if (args.additionalArgsSet && lambda > 0.001f)
	{
		float dx = args.additionalArgs.oldPosition.x[p[0]] - args.x[p[0]];
		float dy = args.additionalArgs.oldPosition.y[p[0]] - args.y[p[0]];
		float dz = args.additionalArgs.oldPosition.z[p[0]] - args.z[p[0]];

		float w1 = dx * s.normal[0] + dy * s.normal[1] + dz * s.normal[2];
		float w2 = s.normal[0] * s.normal[0] + s.normal[1] * s.normal[1] + s.normal[2] * s.normal[2];
		float w = w1 / w2;

		float tmpx = w * s.normal[0];
		float tmpy = w * s.normal[1];
		float tmpz = w * s.normal[2];

		float strength = 5.f;

		float p1 = (dx - tmpx) * strength;
		float p2 = (dy - tmpy) * strength;
		float p3 = (dz - tmpz) * strength;

		float lsq = p1 * p1 + p2 * p2 + p3 * p3;

		if (lsq < muS * muS * r * r)
		{
			atomicAdd(args.dx + p[0], p1);
			atomicAdd(args.dy + p[0], p2);
			atomicAdd(args.dz + p[0], p3);
		}
		else
		{
			float l = sqrt(lsq);
			float coeff = fminf(muD * r / l, 1);

			atomicAdd(args.dx + p[0], coeff * p1);
			atomicAdd(args.dy + p[0], coeff * p2);
			atomicAdd(args.dz + p[0], coeff * p3);
		}
	}
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
