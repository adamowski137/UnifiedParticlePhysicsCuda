#pragma once
#include "../Constraint/DistanceConstraint/DistanceConstraint.cuh"
#include <vector>


class Cloth
{
	static DistanceConstraint* dev_constraints;
	static int nConstraints;
public:
	static void initClothSimulation(int particleH, int particleW, float d,
		float x_top_left, float y_top_left, float z_top_left,
		float* x, float* y, float* z);

	~Cloth();

	static inline std::pair<DistanceConstraint*, int> getConstraints() { return std::make_pair(dev_constraints, nConstraints); }
};
