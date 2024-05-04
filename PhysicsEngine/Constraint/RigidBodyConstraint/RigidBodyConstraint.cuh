#include "../Constraint.cuh"

class RigidBodyConstraint : public Constraint
{

public:
	int* p;
	RigidBodyConstraint init(float* x, float* y, float *z, float* m, int* p, int n, ConstraintLimitType type, float compliance = 0.f);
	~RigidBodyConstraint();

	void calculateShapeCovariance(float* x, float* y, float* z);
	void calculatePositionChange(float* x, float* y, float* z, float* dx, float* dy, float* dz);
private:
	float* rx;
	float* ry;
	float* rz;
	float* decompostion;
	float cx, cy, cz;
};