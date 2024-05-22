#include "RigidBody.cuh"
#include <device_launch_parameters.h>
#include "../../PhysicsEngine/Constants.hpp"
#include "../../PhysicsEngine/GpuErrorHandling.hpp"
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

__global__ void initializePositionsKern(int nParticles, float* x, float* y, float* z, int dim, int nOffset, float xOffset, float yOffset, float zOffset)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;

	int xIndex = index % dim;
	int yIndex = (index / dim) % dim;
	int zIndex = index / (dim * dim);

	x[index + nOffset] = xIndex * ((PARTICLERADIUS - 0.1f) * 2) + xOffset;
	y[index + nOffset] = yIndex * ((PARTICLERADIUS - 0.1f) * 2) + yOffset;
	z[index + nOffset] = zIndex * ((PARTICLERADIUS - 0.1f) * 2) + zOffset;
}

void RigidBody::addRigidBodySquare(float* x, float* y, float* z, float* invmass, int startIdx, int n, float xOffset, float yOffset, float zOffset, int* phase, int phaseIdx)
{
	int nParticles = n * n * n;
	int threads = 32;
	int blocks = (nParticles + threads - 1) / threads;

	initializePositionsKern << <blocks, threads >> > (nParticles, x, y, z, n, startIdx, xOffset, yOffset, zOffset);
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
