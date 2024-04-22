#include "Cloth.hpp"
#include <cmath>
#include <vector>
#include "../GpuErrorHandling.hpp"

DistanceConstraint* Cloth::dev_constraints;
int Cloth::nConstraints;

void Cloth::initClothSimulation(int particleH, int particleW, float d, 
	float x_top_left, float y_top_left, float z_top_left,
	float* x, float* y, float* z)
{
	std::vector<DistanceConstraint> constraints;
	float d_across = d * sqrtf(2);

	std::vector<float> x_cpu(particleH * particleW), y_cpu(particleH * particleW), z_cpu(particleH * particleW);


	// set positions
	for (int i = 0; i < particleH; i++)
	{
		for (int j = 0; j < particleW; j++)
		{
			x_cpu[i * particleW + j] = x_top_left + j * d;
			y_cpu[i * particleW + j] = y_top_left - i * d;
			z_cpu[i * particleW + j] = z_top_left;

			if (j < particleW - 1)
				constraints.push_back(DistanceConstraint().init(d, i * particleW + j, i * particleW + j + 1, ConstraintLimitType::EQ, 20.f));

			if(i > 0)
				constraints.push_back(DistanceConstraint().init(d, i * particleW + j, (i - 1) * particleW + j, ConstraintLimitType::EQ, 20.f));

			if (j < particleW - 1 && i > 0)
			{
				constraints.push_back(DistanceConstraint().init(d_across, i * particleW + j, (i - 1) * particleW + j + 1, ConstraintLimitType::EQ, 20.f));
			}
		}
	}

	gpuErrchk(cudaMemcpy(x, x_cpu.data(), sizeof(float) * x_cpu.size(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(y, y_cpu.data(), sizeof(float) * y_cpu.size(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(z, z_cpu.data(), sizeof(float) * z_cpu.size(), cudaMemcpyHostToDevice));

	cudaMalloc((void**)&dev_constraints, sizeof(DistanceConstraint) * constraints.size());
	cudaMemcpy(dev_constraints, constraints.data(), sizeof(DistanceConstraint) * constraints.size(), cudaMemcpyHostToDevice);

	nConstraints = constraints.size();
}

Cloth::~Cloth()
{
	cudaFree(dev_constraints);
}
