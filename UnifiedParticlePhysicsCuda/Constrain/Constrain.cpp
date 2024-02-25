#include "Constrain.hpp"

Constrain::Constrain(int n, float k, bool equality) : n{n}, k{k}, equality{equality}
{
}

Constrain::~Constrain()
{
}

float Constrain::operator()(int* idx, float* x, float* y, float* z)
{}
