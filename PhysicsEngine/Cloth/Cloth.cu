#include "Cloth.cuh"
#include <cmath>
#include <vector>
#include "../GpuErrorHandling.hpp"
#include "../Config/Config.hpp"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#define GL_FLOAT 0x1406

void Cloth::createMesh(int W, int H, std::vector<float> x_cpu, std::vector<float> y_cpu, std::vector<float> z_cpu)
{
	std::vector<float> vertexData = std::vector<float>(x_cpu.size() * 3);
	std::vector<unsigned int> indicies;
	for (int i = 0; i < x_cpu.size(); i++)
	{
		vertexData[3 * i + 0] = x_cpu[i];
		vertexData[3 * i + 1] = y_cpu[i];
		vertexData[3 * i + 2] = z_cpu[i];
	}

	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			if (i < H - 1)
			{
				indicies.push_back(i * W + j);
				indicies.push_back((i + 1) * W + j);

				if (j < W - 1)
				{
					indicies.push_back((i + 1) * W + j);
					indicies.push_back(i * W + (j + 1));

				}

			}

			if (j < W - 1)
			{
				indicies.push_back(i * W + j);
				indicies.push_back(i * W + (j + 1));
			}


			
		}
	}



	this->clothMesh.generate(vertexData, indicies, { {3, GL_FLOAT} });
}

void Cloth::initClothSimulation_simple(Cloth& cloth, int H, int W, float d,
	float x_top_left, float y_top_left, float z_top_left,
	float* x, float* y, float* z, int* phase, ClothOrientation orientation)
{
	float d_across = d * sqrtf(2);

	std::vector<float> x_cpu(H * W), y_cpu(H * W), z_cpu(H * W);


	// set vertexData
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			switch (orientation)
			{
			case ClothOrientation::XY_PLANE:
				x_cpu[i * W + j] = x_top_left + j * d;
				y_cpu[i * W + j] = y_top_left - i * d;
				z_cpu[i * W + j] = z_top_left;
				break;
			case ClothOrientation::XZ_PLANE:
				x_cpu[i * W + j] = x_top_left + j * d;
				y_cpu[i * W + j] = y_top_left;
				z_cpu[i * W + j] = z_top_left - i * d;
				break;
			case ClothOrientation::YZ_PLANE:
				x_cpu[i * W + j] = x_top_left;
				y_cpu[i * W + j] = y_top_left + j * d;
				z_cpu[i * W + j] = z_top_left - i * d;
				break;
			default:
				break;
			}


			if (j < W - 1)
				cloth.constraints.push_back(DistanceConstraint().init(d, i * W + j, i * W + j + 1, ConstraintLimitType::EQ, GlobalEngineConfig::config.K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING));

			if(i > 0)
				cloth.constraints.push_back(DistanceConstraint().init(d, i * W + j, (i - 1) * W + j, ConstraintLimitType::EQ, GlobalEngineConfig::config.K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING));

			if (j < W - 1 && i > 0)
			{
				cloth.constraints.push_back(DistanceConstraint().init(d_across, i * W + j, (i - 1) * W + j + 1, ConstraintLimitType::EQ, GlobalEngineConfig::config.K_DISTANCE_CONSTRAINT_CLOTH_BENDING));
			}
		}
	}

	auto phase_ptr = thrust::device_pointer_cast(phase);
	thrust::fill(phase_ptr, phase_ptr + W * H, 99);

	gpuErrchk(cudaMemcpy(x, x_cpu.data(), sizeof(float) * x_cpu.size(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(y, y_cpu.data(), sizeof(float) * y_cpu.size(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(z, z_cpu.data(), sizeof(float) * z_cpu.size(), cudaMemcpyHostToDevice));

	cloth.createMesh(W, H, x_cpu, y_cpu, z_cpu);
}

void Cloth::initClothSimulation_LRA(Cloth& cloth, int H, int W, float d,
	float x_top_left, float y_top_left, float z_top_left,
	float* x, float* y, float* z, int* phase, ClothOrientation orientation, std::set<int> attachedParticles)
{
	float d_across = d * sqrtf(2);

	std::vector<float> x_cpu(H * W), y_cpu(H * W), z_cpu(H * W);


	// set vertexData
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			switch (orientation)
			{
			case ClothOrientation::XY_PLANE:
				x_cpu[i * W + j] = x_top_left + j * d;
				y_cpu[i * W + j] = y_top_left - i * d;
				z_cpu[i * W + j] = z_top_left;
				break;
			case ClothOrientation::XZ_PLANE:
				x_cpu[i * W + j] = x_top_left + j * d;
				y_cpu[i * W + j] = y_top_left;
				z_cpu[i * W + j] = z_top_left - i * d;
				break;
			case ClothOrientation::YZ_PLANE:
				x_cpu[i * W + j] = x_top_left;
				y_cpu[i * W + j] = y_top_left + j * d;
				z_cpu[i * W + j] = z_top_left - i * d;
				break;
			default:
				break;
			}


			if (j < W - 1)
				cloth.constraints.push_back(DistanceConstraint().init(d, i * W + j, i * W + j + 1, ConstraintLimitType::EQ, GlobalEngineConfig::config.K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING, false));

			if (i > 0)
				cloth.constraints.push_back(DistanceConstraint().init(d, i * W + j, (i - 1) * W + j, ConstraintLimitType::EQ, GlobalEngineConfig::config.K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING, false));

			if (j < W - 1 && i > 0)
			{
				cloth.constraints.push_back(DistanceConstraint().init(d_across, i * W + j, (i - 1) * W + j + 1, ConstraintLimitType::EQ, GlobalEngineConfig::config.K_DISTANCE_CONSTRAINT_CLOTH_BENDING, false));
			}

			// LRA
			if (attachedParticles.find(i * W + j) == attachedParticles.end())
			{
				for (int particleIdx : attachedParticles)
				{
					int att_i = particleIdx / W;
					int att_j = particleIdx % W;

					float dist = sqrtf((att_i - i) * d * (att_i - i) * d + (att_j - j) * d * (att_j - j) * d);
					cloth.constraints.push_back(DistanceConstraint().init(dist, particleIdx, i * W + j, ConstraintLimitType::EQ, GlobalEngineConfig::config.K_DISTANCE_CONSTRAINT_CLOTH_STRETCHING / 50, false));
				}
			}
		}
	}

	auto phase_ptr = thrust::device_pointer_cast(phase);
	//thrust::fill(phase_ptr, phase_ptr + W * H, 99);

	gpuErrchk(cudaMemcpy(x, x_cpu.data(), sizeof(float) * x_cpu.size(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(y, y_cpu.data(), sizeof(float) * y_cpu.size(), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(z, z_cpu.data(), sizeof(float) * z_cpu.size(), cudaMemcpyHostToDevice));

	cloth.createMesh(W, H, x_cpu, y_cpu, z_cpu);
}
