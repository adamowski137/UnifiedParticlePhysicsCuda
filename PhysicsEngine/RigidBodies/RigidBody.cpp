#include "RigidBody.hpp"

std::vector<RigidBodyConstraint*> RigidBody::constraints{};


void RigidBody::initRigidBodySimulation(float* x, float* y, float* z, float* invmass, std::vector<int> points)
{
	constraints.push_back(new RigidBodyConstraint{ x, y, z, invmass, points.data(), (int)points.size(), ConstraintLimitType::EQ, 1.0f });
}

RigidBody::~RigidBody()
{
	for (auto& c : constraints)
	{
		delete c;
	}
}
