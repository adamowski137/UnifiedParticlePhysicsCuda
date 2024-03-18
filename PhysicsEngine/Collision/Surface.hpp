#pragma once
#include <cmath>

struct Surface
{
	float a, b, c, d;
	float normal[3];
	Surface(float a, float b, float c, float d)
	{
		this->a = a;
		this->b = b;
		this->c = c;
		this->d = d;

		float len = sqrtf(a * a + b * b + c * c);
		normal[0] = a / len;
		normal[1] = b / len;
		normal[2] = c / len;
	}
};
