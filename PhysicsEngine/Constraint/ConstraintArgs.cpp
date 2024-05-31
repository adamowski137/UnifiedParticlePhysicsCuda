#include "ConstraintArgs.hpp"

void ConstraintArgsBuilder::initBase(
	float* x, float* y, float* z,
	int* SDF_mode, float* SDF_value, float* SDF_normal_x, float* SDF_normal_y, float* SDF_normal_z,
	float* invmass)
{
	args.x = x;
	args.y = y;
	args.z = z;
	args.SDF_mode = SDF_mode;
	args.SDF_value = SDF_value;
	args.SDF_normal_x = SDF_normal_x;
	args.SDF_normal_y = SDF_normal_y;
	args.SDF_normal_z = SDF_normal_z;
	args.invmass = invmass;
}

void ConstraintArgsBuilder::addOldPosition(float* x, float* y, float* z)
{
	args.additionalArgsSet = true;
	args.additionalArgs.oldPosition.x = x;
	args.additionalArgs.oldPosition.y = y;
	args.additionalArgs.oldPosition.z = z;
}

void ConstraintArgsBuilder::enableAdditionalArgs()
{
	this->args.additionalArgsSet = true;
}

void ConstraintArgsBuilder::disableAdditionalArgs()
{
	this->args.additionalArgsSet = false;
}

void ConstraintArgsBuilder::clear()
{
	args = {};
}

ConstraintArgs ConstraintArgsBuilder::build()
{
	return args;
}
