#pragma once

struct OldPosition
{
	float* x;
	float* y;
	float* z;
};

union AdditionalArgs
{
	OldPosition oldPosition;
};

struct ConstraintArgs
{
	float* x;
	float* y;
	float* z;
	float* new_x;
	float* new_y;
	float* new_z;
	float* dx;
	float* dy;
	float* dz;
	int* SDF_mode;
	float* SDF_value;
	float* SDF_normal_x;
	float* SDF_normal_y;
	float* SDF_normal_z;
	float* invmass;
	int* nConstraintsPerParticle;
	float dt;
	bool additionalArgsSet;
	AdditionalArgs additionalArgs;
};




class ConstraintArgsBuilder
{
public:
	ConstraintArgs args;
	void initBase(
		float* x, float* y, float* z,
		int* SDF_mode, float* SDF_value, float* SDF_normal_x, float* SDF_normal_y, float* SDF_normal_z,
		float* invmass);
	void addOldPosition(float* x, float* y, float* z);
	void enableAdditionalArgs();
	void disableAdditionalArgs();
	void clear();
	ConstraintArgs build();
};
