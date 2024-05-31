#pragma once
#include "../Constraint/RigidBodyConstraint/RigidBodyConstraint.cuh"
#include <vector>


class RigidBody
{
	std::vector<RigidBodyConstraint*> constraints;
public:
	void addRigidBodySquare(float* x, float* y, float* z,
		int* dev_SDF_mode, float* dev_SDF_value, float* dev_SDF_normal_x, float* dev_SDF_normal_y, float* dev_SDF_normal_z,
		float* invmass, int startIdx, int n,
		float xOffset, float yOffset, float zOffset, int* phase, int phaseIdx);
	void initRigidBodySimulation(float* x, float* y, float* z, float* invmass, std::vector<int> points);
	inline std::vector<RigidBodyConstraint*> getConstraints() { return constraints; }

	~RigidBody();
};
