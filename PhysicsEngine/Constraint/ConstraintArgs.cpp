#include "ConstraintArgs.hpp"

void ConstraintArgsBuilder::initBase(float* x, float* y, float* z, float* dx, float* dy, float* dz, float* invmass, int* nConstraintsPerParticle, float dt)
{
	args.x = x;
	args.y = y;
	args.z = z;
	args.dx = dx;
	args.dy = dy;
	args.dz = dz;
	args.invmass = invmass;
	args.nConstraintsPerParticle = nConstraintsPerParticle;
	args.dt = dt;
}

void ConstraintArgsBuilder::addOldPosition(float* x, float* y, float* z)
{
	args.additionalArgsSet = true;
	args.additionalArgs.oldPosition.x = x;
	args.additionalArgs.oldPosition.y = y;
	args.additionalArgs.oldPosition.z = z;
}

void ConstraintArgsBuilder::clear()
{
	args = {};
}

ConstraintArgs ConstraintArgsBuilder::build()
{
	return args;
}
