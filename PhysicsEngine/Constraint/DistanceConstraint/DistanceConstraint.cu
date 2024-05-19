#include "DistanceConstraint.cuh"
#include <cmath>
#include "../Constraint.cuh"

__host__ __device__ DistanceConstraint DistanceConstraint::init(float d, int p1, int p2, ConstraintLimitType type, float compliance)
{
	((Constraint*)this)->init(2, compliance, type);
	this->p[0] = p1;
	this->p[1] = p2;
	this->d = d;
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

__device__ void DistanceConstraint::directSolve(float* x, float* y, float* z, float* dx, float* dy, float* dz, float* invmass, int* nConstraintsPerParticle, float dt)
{
	float distX = (x[p[0]] - x[p[1]]);
	float distY = (y[p[0]] - y[p[1]]);
	float distZ = (z[p[0]] - z[p[1]]);

	float dist = sqrt(distX * distX + distY * distY + distZ * distZ);
	float C = (dist - d);
	float invw = 1.f / (invmass[p[0]] + invmass[p[1]]);

	float lambda = -C * k * invw;

	lambda = min(max(lambda, cMin), cMax);

	float coeff_p0 = invmass[p[0]] * lambda / dist;
	float coeff_p1 = -invmass[p[1]] * lambda / dist;


	atomicAdd(dx + p[0], coeff_p0 * distX);
	atomicAdd(dy + p[0], coeff_p0 * distY);
	atomicAdd(dz + p[0], coeff_p0 * distZ);

	atomicAdd(dx + p[1], coeff_p1 * distX);
	atomicAdd(dy + p[1], coeff_p1 * distY);
	atomicAdd(dz + p[1], coeff_p1 * distZ);

	atomicAdd(nConstraintsPerParticle + p[0], 1);
	atomicAdd(nConstraintsPerParticle + p[1], 1);
}

__host__ void DistanceConstraint::directSolve_cpu(float* x, float* y, float* z, float* invmass)
{
	float distX = (x[p[0]] - x[p[1]]);
	float distY = (y[p[0]] - y[p[1]]);
	float distZ = (z[p[0]] - z[p[1]]);

	float dist = sqrt(distX * distX + distY * distY + distZ * distZ);
	float C = (dist - d);
	float invw = 1.f / (invmass[p[0]] + invmass[p[1]]);

	float lambda = - C * invw * k;

	lambda = std::fmin(std::fmax(lambda, cMin), cMax);

	float coeff_p0 = invmass[p[0]] * lambda / dist;
	float coeff_p1 = -invmass[p[1]] * lambda / dist;

	x[p[0]] += coeff_p0 * distX;
	y[p[0]] += coeff_p0 * distY;
	z[p[0]] += coeff_p0 * distZ;

	x[p[1]] += coeff_p1 * distX;
	y[p[1]] += coeff_p1 * distY;
	z[p[1]] += coeff_p1 * distZ;
}
