#pragma once 

#include <vector>

class RigidBody
{
public:
	RigidBody(std::vector<int> points, float* x, float* y, float* z, float* invmass);
	~RigidBody();
	int n;
	int* points;
	float* rx;
	float* ry;
	float* rz;
	float* A;
	const int matrixSize = 9;

private:
	void calculateRadius(float* x, float* y, float* z, float* invmass);
	void calculateA(float* invmass);
};