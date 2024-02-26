#pragma once
#include "../Constrain.hpp"
#include <cuda_runtime.h>

class FloorConstrain
{
public:
	FloorConstrain();
	void fillJacobian(int particles, int constrains, float* jacobian);
};
__global__ void fillJacobianKern(int particles, int size, float* jacobian);
