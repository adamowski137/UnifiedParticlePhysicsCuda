#include "../Constrain.hpp"

class DistanceConstrain : public Constrain
{
public:
	DistanceConstrain(float k, float d);
	float virtual operator()(int* index, float* x, float* y, float* z);
private:
	float d;
};