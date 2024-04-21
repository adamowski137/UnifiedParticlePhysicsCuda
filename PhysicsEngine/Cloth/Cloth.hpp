#pragma once

class Cloth
{
public:
	static void initClothSimulation(int particleH, int particleW, float d,
		float x_top_left, float y_top_left, float z_top_left,
		float* x, float* y, float* z);
};
