#pragma once
#include "../Constraint/DistanceConstraint/DistanceConstraint.cuh"
#include <vector>
#include <set>
#include "../../GUI/Renderer/RenderInfo.hpp"


enum class ClothOrientation {XY_PLANE, XZ_PLANE, YZ_PLANE};

class Cloth
{
	std::vector<DistanceConstraint> constraints;

	void createMesh(int W, int H,
		std::vector<float> x_cpu, std::vector<float> y_cpu, std::vector<float> z_cpu);

public:
	static void initClothSimulation_simple(Cloth& cloth, int H, int W, float d,
		float x_top_left, float y_top_left, float z_top_left,
		float* x, float* y, float* z, int* phase, ClothOrientation orientation);

	static void initClothSimulation_LRA(Cloth& cloth, int H, int W, float d,
		float x_top_left, float y_top_left, float z_top_left,
		float* x, float* y, float* z, int* phase, ClothOrientation orientation, std::set<int> attachedParticles);


	RenderInfo clothMesh;
	inline std::pair<DistanceConstraint*, int> getConstraints() { return std::make_pair(constraints.data(), constraints.size()); }
};
