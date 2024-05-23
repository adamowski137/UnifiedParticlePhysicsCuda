#pragma once
#include "../Constraint/DistanceConstraint/DistanceConstraint.cuh"
#include <vector>
#include <set>


enum class ClothOrientation {XY_PLANE, XZ_PLANE, YZ_PLANE};

class Cloth
{
	std::vector<DistanceConstraint> constraints;
public:
	static void initClothSimulation_simple(Cloth& cloth, int particleH, int particleW, float d,
		float x_top_left, float y_top_left, float z_top_left,
		float* x, float* y, float* z, int* phase, ClothOrientation orientation);

	static void initClothSimulation_LRA(Cloth& cloth, int particleH, int particleW, float d,
		float x_top_left, float y_top_left, float z_top_left,
		float* x, float* y, float* z, int* phase, ClothOrientation orientation, std::set<int> attachedParticles);

	inline std::pair<DistanceConstraint*, int> getConstraints() { return std::make_pair(constraints.data(), constraints.size()); }
};
