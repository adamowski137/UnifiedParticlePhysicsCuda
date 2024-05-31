#include "RigidBody.cuh"
#include <device_launch_parameters.h>
#include "../../PhysicsEngine/Constants.hpp"
#include "../../PhysicsEngine/GpuErrorHandling.hpp"
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

__global__ void initializePositionsKern(int nParticles, float* x, float* y, float* z, 
	int* SDF_mode, float* SDF_value, float* SDF_normal_x, float* SDF_normal_y, float* SDF_normal_z,
	int dim, int nOffset, float xOffset, float yOffset, float zOffset)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;

	int xIndex = index % dim;
	int yIndex = (index / dim) % dim;
	int zIndex = index / (dim * dim);

	x[index + nOffset] = xIndex * ((PARTICLERADIUS - 0.1f) * 2) + xOffset;
	y[index + nOffset] = yIndex * ((PARTICLERADIUS - 0.1f) * 2) + yOffset;
	z[index + nOffset] = zIndex * ((PARTICLERADIUS - 0.1f) * 2) + zOffset;

	float sdf_x_offset = (float)xIndex - (float)dim / 2;
	float sdf_y_offset = (float)yIndex - (float)dim / 2;
	float sdf_z_offset = (float)zIndex - (float)dim / 2;

	float sdf_offset_max = max(abs(sdf_x_offset), max(abs(sdf_y_offset), abs(sdf_z_offset)));
	if (abs(sdf_x_offset) < sdf_offset_max) sdf_x_offset = 0;
	if (abs(sdf_y_offset) < sdf_offset_max) sdf_y_offset = 0;
	if (abs(sdf_z_offset) < sdf_offset_max) sdf_z_offset = 0;

	SDF_mode[index + nOffset] = 0;
	//SDF_mode[index + nOffset] = 1 + (xIndex == 0 | xIndex == dim - 1 | yIndex == 0 | yIndex == dim - 1 | zIndex == 0 | zIndex == dim - 1);
	SDF_value[index + nOffset] = -1 - min((float)dim / 2 - abs(sdf_x_offset), min((float)dim / 2 - abs(sdf_y_offset), (float)dim / 2 - abs(sdf_z_offset)));
	float len = sqrt(sdf_x_offset * sdf_x_offset + sdf_y_offset * sdf_y_offset + sdf_z_offset * sdf_z_offset);
	SDF_normal_x[index + nOffset] = sdf_x_offset / len;
	SDF_normal_y[index + nOffset] = sdf_y_offset / len;
	SDF_normal_z[index + nOffset] = sdf_z_offset / len;
	
}

void RigidBody::addRigidBodySquare(float* x, float* y, float* z, 
	int* dev_SDF_mode, float* dev_SDF_value, float* dev_SDF_normal_x, float* dev_SDF_normal_y, float* dev_SDF_normal_z,
	float* invmass, int startIdx, int n, float xOffset, float yOffset, float zOffset, int* phase, int phaseIdx)
{
	int nParticles = n * n * n;
	int threads = 32;
	int blocks = (nParticles + threads - 1) / threads;

	initializePositionsKern << <blocks, threads >> > (nParticles, x, y, z,
		dev_SDF_mode, dev_SDF_value, dev_SDF_normal_x, dev_SDF_normal_y, dev_SDF_normal_z,
		n, startIdx, xOffset, yOffset, zOffset);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	auto phase_ptr = thrust::device_pointer_cast(phase);
	thrust::fill(phase_ptr + startIdx, phase_ptr + startIdx + nParticles, phaseIdx);

	std::vector<int> points(nParticles, startIdx);
	for (int i = 0; i < nParticles; i++)
		points[i] += i;
	initRigidBodySimulation(x, y, z, invmass, points);
}

void RigidBody::initRigidBodySimulation(float* x, float* y, float* z, float* invmass, std::vector<int> points)
{
	constraints.push_back(new RigidBodyConstraint{ x, y, z, invmass, points.data(), (int)points.size(), ConstraintLimitType::EQ, 1.0f });
}

RigidBody::~RigidBody()
{
	for (auto& c : constraints)
	{
		delete c;
	}
}
