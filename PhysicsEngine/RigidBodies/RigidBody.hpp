#pragma once
#include "../Constraint/RigidBodyConstraint/RigidBodyConstraint.cuh"
#include <vector>


class RigidBody
{
	static std::vector<RigidBodyConstraint*> constraints;
public:

	static void initRigidBodySimulation(float* x, float* y, float* z, float* invmass, std::vector<int> points);

	static inline std::vector<RigidBodyConstraint*> getConstraints() { return constraints; }

	~RigidBody();
};
