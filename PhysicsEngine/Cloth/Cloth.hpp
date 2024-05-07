#pragma once
#include "../Constraint/DistanceConstraint/DistanceConstraint.cuh"
#include <vector>


class Cloth
{
	static std::vector<DistanceConstraint> constraints;
public:
	static void initClothSimulation(int particleH, int particleW, float d,
		float x_top_left, float y_top_left, float z_top_left,
		float* x, float* y, float* z);

	static inline std::pair<DistanceConstraint*, int> getConstraints() { return std::make_pair(constraints.data(), constraints.size()); }
};
