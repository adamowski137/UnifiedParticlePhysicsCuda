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
	float* dx;
	float* dy;
	float* dz;
	float* invmass;
	int* nConstraintsPerParticle;
	float dt;
	bool additionalArgsSet;
	AdditionalArgs additionalArgs;
};




class ConstraintArgsBuilder
{
private:
	ConstraintArgs args;
public:
	void initBase(float* x, float* y, float* z, float* dx, float* dy, float* dz, float* invmass, int* nConstraintsPerParticle, float dt);
	void addOldPosition(float* x, float* y, float* z);
	void clear();
	ConstraintArgs build();
};
