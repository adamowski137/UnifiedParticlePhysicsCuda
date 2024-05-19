#pragma once
#include "../Constraint/RigidBodyConstraint/RigidBodyConstraint.cuh"
#include <vector>


class RigidBody
{
	std::vector<RigidBodyConstraint*> constraints;
public:
	void addRigidBodySquare(float* x, float* y, float* z,
		float* invmass, int startIdx, int n,
		float xOffset, float yOffset, float zOffset);
	void initRigidBodySimulation(float* x, float* y, float* z, float* invmass, std::vector<int> points);
	inline std::vector<RigidBodyConstraint*> getConstraints() { return constraints; }

	~RigidBody();
};
