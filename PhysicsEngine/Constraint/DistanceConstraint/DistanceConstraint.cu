#include "DistanceConstraint.cuh"
#include <cmath>
#include "../Constraint.cuh"

__host__ __device__ DistanceConstraint DistanceConstraint::init(float d, int p1, int p2, ConstraintLimitType type, float k, bool apply_friction)
{
	((Constraint*)this)->init(2, k, type);
	this->p[0] = p1;
	this->p[1] = p2;
	this->d = d;
	this->apply_friction = apply_friction;
	return *this;
}

__host__ __device__ float DistanceConstraint::operator()(float* x, float* y, float* z)
{
	float distX = (x[p[0]] - x[p[1]]) * (x[p[0]] - x[p[1]]);
	float distY = (y[p[0]] - y[p[1]]) * (y[p[0]] - y[p[1]]);
	float distZ = (z[p[0]] - z[p[1]]) * (z[p[0]] - z[p[1]]);

	float C = (sqrtf(distX + distY + distZ) - d);
	return C;
}

__host__ __device__ void DistanceConstraint::positionDerivative(float* x, float* y, float* z, float* jacobian, int nParticles, int index)
{
	float distX = (x[p[0]] - x[p[1]]) * (x[p[0]] - x[p[1]]);
	float distY = (y[p[0]] - y[p[1]]) * (y[p[0]] - y[p[1]]);
	float distZ = (z[p[0]] - z[p[1]]) * (z[p[0]] - z[p[1]]);

	float d = sqrtf(distX + distY + distZ);
	//d = 1;

	int idx1 = index * 3 * nParticles + 3 * p[0];
	int idx2 = index * 3 * nParticles + 3 * p[1];

	float dCx = (x[p[0]] - x[p[1]]) / d;
	float dCy = (y[p[0]] - y[p[1]]) / d;
	float dCz = (z[p[0]] - z[p[1]]) / d;

	jacobian[idx1 + 0] = dCx;
	jacobian[idx1 + 1] = dCy;
	jacobian[idx1 + 2] = dCz;

	jacobian[idx2 + 0] = -dCx;
	jacobian[idx2 + 1] = -dCy;
	jacobian[idx2 + 2] = -dCz;
}

__device__ void DistanceConstraint::directSolve(ConstraintArgs args)
{
	float distX = (args.x[p[0]] - args.x[p[1]]);
	float distY = (args.y[p[0]] - args.y[p[1]]);
	float distZ = (args.z[p[0]] - args.z[p[1]]);

	float dist = sqrt(distX * distX + distY * distY + distZ * distZ);
	float C = (dist - d);
	float invw = 1.f / (args.invmass[p[0]] + args.invmass[p[1]]);

	float lambda = -C * k * invw;

	lambda = min(max(lambda, cMin), cMax);

	float coeff_p0 = args.invmass[p[0]] * lambda / dist;
	float coeff_p1 = -args.invmass[p[1]] * lambda / dist;

	

	atomicAdd(args.dx + p[0], coeff_p0 * distX);
	atomicAdd(args.dy + p[0], coeff_p0 * distY);
	atomicAdd(args.dz + p[0], coeff_p0 * distZ);

	atomicAdd(args.dx + p[1], coeff_p1 * distX);
	atomicAdd(args.dy + p[1], coeff_p1 * distY);
	atomicAdd(args.dz + p[1], coeff_p1 * distZ);

	atomicAdd(args.nConstraintsPerParticle + p[0], 1);
	atomicAdd(args.nConstraintsPerParticle + p[1], 1);

	if (apply_friction && args.additionalArgsSet && lambda > 0.001f)
	{

		float dx = (args.additionalArgs.oldPosition.x[p[0]] - args.x[p[0]]) - (args.additionalArgs.oldPosition.x[p[1]] - args.x[p[1]]);
		float dy = (args.additionalArgs.oldPosition.y[p[0]] - args.y[p[0]]) - (args.additionalArgs.oldPosition.y[p[1]] - args.y[p[1]]);
		float dz = (args.additionalArgs.oldPosition.z[p[0]] - args.z[p[0]]) - (args.additionalArgs.oldPosition.z[p[1]] - args.z[p[1]]);

		float len = sqrt(distX * distX + distY * distY + distZ * distZ);

		float nx = distX / len;
		float ny = distY / len;
		float nz = distZ / len;

		float w = dx * nx + dy * ny + dz * nz;

		float tmpx = w * nx;
		float tmpy = w * ny;
		float tmpz = w * nz;

		float p1 = dx - tmpx;
		float p2 = dy - tmpy;
		float p3 = dz - tmpz;

		float lsq = p1 * p1 + p2 * p2 + p3 * p3;

		float denom = 1 / (1 / (args.invmass[p[0]]) + 1 / (args.invmass[p[1]]));
		float w1 = (1 / args.invmass[p[0]]) * denom;
		float w2 = -(1 / args.invmass[p[1]]) * denom;

		if (lsq < muS * muS * d * d)
		{
			atomicAdd(args.dx + p[0], w1 * p1);
			atomicAdd(args.dy + p[0], w1 * p2);
			atomicAdd(args.dz + p[0], w1 * p3);
			atomicAdd(args.dx + p[1], w2 * p1);
			atomicAdd(args.dy + p[1], w2 * p2);
			atomicAdd(args.dz + p[1], w2 * p3);
		}
		else
		{
			float l = sqrt(lsq);
			float coeff = fminf(muD * d / l, 1);

			atomicAdd(args.dx + p[0], w1 * coeff * p1);
			atomicAdd(args.dy + p[0], w1 * coeff * p2);
			atomicAdd(args.dz + p[0], w1 * coeff * p3);
			atomicAdd(args.dx + p[1], w2 * coeff * p1);
			atomicAdd(args.dy + p[1], w2 * coeff * p2);
			atomicAdd(args.dz + p[1], w2 * coeff * p3);
		}
	}
}

__host__ __device__ void DistanceConstraint::directSolve_GaussSeidel(ConstraintArgs args)
{
	float distX = (args.x[p[0]] - args.x[p[1]]);
	float distY = (args.y[p[0]] - args.y[p[1]]);
	float distZ = (args.z[p[0]] - args.z[p[1]]);

	float dist = sqrt(distX * distX + distY * distY + distZ * distZ);
	float C = (dist - d);
	float invw = 1.f / (args.invmass[p[0]] + args.invmass[p[1]]);

	float lambda = - C * invw * k;

	lambda = min(max(lambda, cMin), cMax);

	float coeff_p0 = args.invmass[p[0]] * lambda / dist;
	float coeff_p1 = -args.invmass[p[1]] * lambda / dist;

	args.x[p[0]] += coeff_p0 * distX;
	args.y[p[0]] += coeff_p0 * distY;
	args.z[p[0]] += coeff_p0 * distZ;

	args.x[p[1]] += coeff_p1 * distX;
	args.y[p[1]] += coeff_p1 * distY;
	args.z[p[1]] += coeff_p1 * distZ;
}
