#pragma once
#include "../Constrain.cuh"
#include <cuda_runtime.h>

class FloorConstrain : public Constrain
{
public:
	FloorConstrain(int* indexes);
	virtual void fillJacobian(float* jacobianRow);
};
__global__ void fillJacobianKern(int n, float* jacobianRow, int* idx);
