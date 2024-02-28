#pragma once
class Constrain
{
public:
	Constrain(int n, float k, float cMin, float cMax, int* indexes);
	~Constrain();
	virtual void fillJacobian(float* jacobianRow) = 0;
protected:
	int* dev_indexes;
	int n;
	float k;
	float cMin;
	float cMax;
};