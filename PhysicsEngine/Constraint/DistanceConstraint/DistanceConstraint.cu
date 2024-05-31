#include "DistanceConstraint.cuh"
#include <cmath>
#include "../Constraint.cuh"

#define DOT(x0, y0, z0, x1, y1, z1) ((x0 * x1) + (y0 * y1) + (z0 * z1))

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

__host__ __device__ void DistanceConstraint::calculateNormalVector(ConstraintArgs args, float n[3], float* dist)
{
	bool p0_is_sdf = args.SDF_mode[p[0]] != 0;
	bool p1_is_sdf = args.SDF_mode[p[1]] != 0;

	bool useSDF = false;
	int SDF_particle_index = 0;

	float xij_X = (args.x[p[0]] - args.x[p[1]]);
	float xij_Y = (args.y[p[0]] - args.y[p[1]]);
	float xij_Z = (args.z[p[0]] - args.z[p[1]]);

	*dist = sqrt(xij_X * xij_X + xij_Y * xij_Y + xij_Z * xij_Z);

	xij_X /= *dist;
	xij_Y /= *dist;
	xij_Z /= *dist;

	if (p0_is_sdf)
	{
		useSDF = true;
		if (p1_is_sdf)
		{
			SDF_particle_index = abs(args.SDF_value[p[0]]) < abs(args.SDF_value[p[1]]) ? p[0] : p[1];
		}
		else
		{
			SDF_particle_index = p[0];
		}
	}
	else if (p1_is_sdf)
	{
		useSDF = true;
		SDF_particle_index = p[1];
	}
	else
	{
		n[0] = xij_X;
		n[1] = xij_Y;
		n[2] = xij_Z;
	}

	float SDF_sign = SDF_particle_index == p[0] ? -1.f : 1.f;
	if (useSDF)
	{
		n[0] = SDF_sign * args.SDF_normal_x[SDF_particle_index];
		n[1] = SDF_sign * args.SDF_normal_y[SDF_particle_index];
		n[2] = SDF_sign * args.SDF_normal_z[SDF_particle_index];
		// shape border sdf
		if (args.SDF_mode[SDF_particle_index] == 2)
		{
			float dotp = DOT(xij_X, xij_Y, xij_Z, n[0], n[1], n[2]);
			if (dotp < 0)
			{
				n[0] = xij_X - 2 * dotp * n[0];
				n[1] = xij_Y - 2 * dotp * n[1];
				n[2] = xij_Z - 2 * dotp * n[2];
			}
			else
			{
				n[0] = xij_X;
				n[1] = xij_Y;
				n[2] = xij_Z;
			}
		}
	}
}

__device__ void DistanceConstraint::directSolve(ConstraintArgs args)
{
	float n[3], dist;
	calculateNormalVector(args, n, &dist);

	float C = (dist - d);
	float invw = 1.f / (args.invmass[p[0]] + args.invmass[p[1]]);

	float lambda = -C * k * invw;

	lambda = min(max(lambda, cMin), cMax);

	float coeff_p0 = args.invmass[p[0]] * lambda;
	float coeff_p1 = -args.invmass[p[1]] * lambda;

	

	atomicAdd(args.dx + p[0], coeff_p0 * n[0]);
	atomicAdd(args.dy + p[0], coeff_p0 * n[1]);
	atomicAdd(args.dz + p[0], coeff_p0 * n[2]);

	atomicAdd(args.dx + p[1], coeff_p1 * n[0]);
	atomicAdd(args.dy + p[1], coeff_p1 * n[1]);
	atomicAdd(args.dz + p[1], coeff_p1 * n[2]);

	atomicAdd(args.nConstraintsPerParticle + p[0], 1);
	atomicAdd(args.nConstraintsPerParticle + p[1], 1);

	if (apply_friction && args.additionalArgsSet && lambda > 0.001f)
	{

		float dx = (args.additionalArgs.oldPosition.x[p[0]] - args.x[p[0]]) - (args.additionalArgs.oldPosition.x[p[1]] - args.x[p[1]]);
		float dy = (args.additionalArgs.oldPosition.y[p[0]] - args.y[p[0]]) - (args.additionalArgs.oldPosition.y[p[1]] - args.y[p[1]]);
		float dz = (args.additionalArgs.oldPosition.z[p[0]] - args.z[p[0]]) - (args.additionalArgs.oldPosition.z[p[1]] - args.z[p[1]]);

		float w = dx * n[0] + dy * n[1] + dz * n[2];

		float tmpx = w * n[0];
		float tmpy = w * n[1];
		float tmpz = w * n[2];
		
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
