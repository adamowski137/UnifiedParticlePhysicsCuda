#include "Cloth.cuh"
#include <cmath>
#include <vector>
#include "../GpuErrorHandling.hpp"
#include "../Config/Config.hpp"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

void Cloth::initClothSimulation_simple(Cloth& cloth, int particleH, int particleW, float d,
	float x_top_left, float y_top_left, float z_top_left,
	float* x, float* y, float* z, int* phase, ClothOrientation orientation)
{
	float d_across = d * sqrtf(2);

	std::vector<float> x_cpu(particleH * particleW), y_cpu(particleH * particleW), z_cpu(particleH * particleW);


	// set positions
	for (int i = 0; i < particleH; i++)
	{
		for (int j = 0; j < particleW; j++)
		{
			switch (orientation)
			{
			case ClothOrientation::XY_PLANE:
				x_cpu[i * particleW + j] = x_top_left + j * d;
				y_cpu[i * particleW + j] = y_top_left - i * d;
				z_cpu[i * particleW + j] = z_top_left;
				break;
			case ClothOrientation::XZ_PLANE:
				x_cpu[i * particleW + j] = x_top_left + j * d;
				y_cpu[i * particleW + j] = y_top_left;
				z_cpu[i * particleW + j] = z_top_left - i * d;
				break;
			case ClothOrientation::YZ_PLANE:
				x_cpu[i * particleW + j] = x_top_left;
				y_cpu[i * particleW + j] = y_top_left + j * d;
				z_cpu[i * particleW + j] = z_top_left - i * d;
				break;
			default:
				break;
			}


			if (j < particleW - 1)
				cloth.constraints.push_back(DistanceConstraint().init(d, i * particleW + j, i * particleW + j + 1, ConstraintLimitType::EQ, EngineConfig::K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING));

			if(i > 0)
				cloth.constraints.push_back(DistanceConstraint().init(d, i * particleW + j, (i - 1) * particleW + j, ConstraintLimitType::EQ, EngineConfig::K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING));

			if (j < particleW - 1 && i > 0)
			{
				cloth.constraints.push_back(DistanceConstraint().init(d_across, i * particleW + j, (i - 1) * particleW + j + 1, ConstraintLimitType::EQ, EngineConfig::K_DISTANCE_CONSTRAINT_CLOTH_BENDING));
			}
		}
	}

	auto phase_ptr = thrust::device_pointer_cast(phase);
	//thrust::fill(phase_ptr, phase_ptr + particleW * particleH, 99);

	gpuErrchk(cudaMemcpy(x, x_cpu.data(), sizeof(float) * x_cpu.size(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(y, y_cpu.data(), sizeof(float) * y_cpu.size(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(z, z_cpu.data(), sizeof(float) * z_cpu.size(), cudaMemcpyHostToDevice));
}

void Cloth::initClothSimulation_LRA(Cloth& cloth, int particleH, int particleW, float d,
	float x_top_left, float y_top_left, float z_top_left,
	float* x, float* y, float* z, int* phase, ClothOrientation orientation, std::set<int> attachedParticles)
{
	float d_across = d * sqrtf(2);

	std::vector<float> x_cpu(particleH * particleW), y_cpu(particleH * particleW), z_cpu(particleH * particleW);


	// set positions
	for (int i = 0; i < particleH; i++)
	{
		for (int j = 0; j < particleW; j++)
		{
			switch (orientation)
			{
			case ClothOrientation::XY_PLANE:
				x_cpu[i * particleW + j] = x_top_left + j * d;
				y_cpu[i * particleW + j] = y_top_left - i * d;
				z_cpu[i * particleW + j] = z_top_left;
				break;
			case ClothOrientation::XZ_PLANE:
				x_cpu[i * particleW + j] = x_top_left + j * d;
				y_cpu[i * particleW + j] = y_top_left;
				z_cpu[i * particleW + j] = z_top_left - i * d;
				break;
			case ClothOrientation::YZ_PLANE:
				x_cpu[i * particleW + j] = x_top_left;
				y_cpu[i * particleW + j] = y_top_left + j * d;
				z_cpu[i * particleW + j] = z_top_left - i * d;
				break;
			default:
				break;
			}


			if (j < particleW - 1)
				cloth.constraints.push_back(DistanceConstraint().init(d, i * particleW + j, i * particleW + j + 1, ConstraintLimitType::EQ, EngineConfig::K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING, false));

			if (i > 0)
				cloth.constraints.push_back(DistanceConstraint().init(d, i * particleW + j, (i - 1) * particleW + j, ConstraintLimitType::EQ, EngineConfig::K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING, false));

			if (j < particleW - 1 && i > 0)
			{
				cloth.constraints.push_back(DistanceConstraint().init(d_across, i * particleW + j, (i - 1) * particleW + j + 1, ConstraintLimitType::EQ, EngineConfig::K_DISTANCE_CONSTRAINT_CLOTH_BENDING, false));
			}

			// LRA
			if (attachedParticles.find(i * particleW + j) == attachedParticles.end())
			{
				for (int particleIdx : attachedParticles)
				{
					int att_i = particleIdx / particleW;
					int att_j = particleIdx % particleW;

					float dist = sqrtf((att_i - i) * d * (att_i - i) * d + (att_j - j) * d * (att_j - j) * d);
					cloth.constraints.push_back(DistanceConstraint().init(dist, particleIdx, i * particleW + j, ConstraintLimitType::EQ, EngineConfig::K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING / 100, false));
				}
			}
		}
	}

	auto phase_ptr = thrust::device_pointer_cast(phase);
	//thrust::fill(phase_ptr, phase_ptr + particleW * particleH, 99);

	gpuErrchk(cudaMemcpy(x, x_cpu.data(), sizeof(float) * x_cpu.size(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(y, y_cpu.data(), sizeof(float) * y_cpu.size(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(z, z_cpu.data(), sizeof(float) * z_cpu.size(), cudaMemcpyHostToDevice));
}
