#include "SurfaceCollisionFinder.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../GpuErrorHandling.hpp"

#include "thrust/device_ptr.h"
#include "thrust/device_vector.h"
#include "thrust/scan.h"
#include "../Constants.hpp"


__global__ void findSurfaceCollisions(
	int nParticles, int nSurfaces,
	int* hits, Surface* surfaces,
	float* x, float* y, float* z, float r)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= nParticles) return;

	for (int i = 0; i < nSurfaces; i++)
	{
		Surface s = surfaces[i];
		float v1 = s.a * x[index] + s.b * y[index] + s.c * z[index] + s.d;
		float sign = v1 < 0 ? 1 : -1;

		float v2 = s.a * (x[index] + sign * s.normal[0] * r) + s.b * (y[index] + sign * s.normal[1] * r) + s.c * (z[index] + sign * s.normal[2] * r) + s.d;

		if (v1 * v2 < 0) // hit
		{
			hits[i * nParticles + index] = 1;
		}
	}
}

__global__ void fillConstraints(int nParticles, int nSurfaces,
	SurfaceConstraint* constraints,
	int* hits, int* hitsSum, Surface* surfaces, float r)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= nParticles) return;

	for (int i = 0; i < nSurfaces; i++)
	{
		if (hits[index + i * nParticles])
		{
			// offset by 1 because we index array starting from 0
			int constraintIndex = hitsSum[index + i * nParticles] - 1;
			constraints[constraintIndex] = SurfaceConstraint(r, index, surfaces[i]);
		}
	}
}

SurfaceCollisionFinder::SurfaceCollisionFinder(std::vector<Surface> surfaces, int nParticles)
{
	nSurfaces = surfaces.size();
	gpuErrchk(cudaMalloc((void**)&dev_hit, sizeof(int) * nParticles * surfaces.size()));
	gpuErrchk(cudaMemset(dev_hit, 0, sizeof(int) * nParticles * surfaces.size()));

	gpuErrchk(cudaMalloc((void**)&dev_surface, sizeof(Surface) * surfaces.size()));
	gpuErrchk(cudaMemcpy(dev_surface, surfaces.data(), sizeof(Surface) * surfaces.size(), cudaMemcpyHostToDevice));
}

SurfaceCollisionFinder::~SurfaceCollisionFinder()
{
	gpuErrchk(cudaFree(dev_hit));
	gpuErrchk(cudaFree(dev_surface));
}

void SurfaceCollisionFinder::setSurfaces(std::vector<Surface> surfaces, int nParticles)
{
	gpuErrchk(cudaFree(dev_hit));
	gpuErrchk(cudaFree(dev_surface));

	nSurfaces = surfaces.size();

	gpuErrchk(cudaMalloc((void**)&dev_hit, sizeof(int) * nParticles * surfaces.size()));
	gpuErrchk(cudaMemset(dev_hit, 0, sizeof(int) * nParticles * surfaces.size()));

	gpuErrchk(cudaMalloc((void**)&dev_surface, sizeof(Surface) * surfaces.size()));
	gpuErrchk(cudaMemcpy(dev_surface, surfaces.data(), sizeof(Surface) * surfaces.size(), cudaMemcpyHostToDevice));

}

std::pair<SurfaceConstraint*, int> SurfaceCollisionFinder::findAndUpdateCollisions(int nParticles, float* x, float* y, float* z)
{
	if (nSurfaces == 0)
		return std::make_pair((SurfaceConstraint*)0, 0);

	int threads = 32;
	int blocks = (nParticles + threads - 1) / threads;

	findSurfaceCollisions << <blocks, threads >> > (nParticles, nSurfaces, dev_hit, dev_surface, x, y, z, PARTICLERADIUS);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::device_vector<int> prefixSum(nParticles * nSurfaces);
	thrust::device_ptr<int> p = thrust::device_pointer_cast<int>(dev_hit);
	thrust::inclusive_scan(p, p + nSurfaces * nParticles, prefixSum.begin());

	int nCollisions = prefixSum[nParticles * nSurfaces - 1];
	SurfaceConstraint* dev_constraints;

	gpuErrchk(cudaMalloc((void**)&dev_constraints, sizeof(SurfaceConstraint) * nCollisions));
	int* dev_hitsSum = thrust::raw_pointer_cast(prefixSum.data());


	fillConstraints << <blocks, threads >> > (nParticles, nSurfaces,
		dev_constraints,
		dev_hit, dev_hitsSum, dev_surface, PARTICLERADIUS / 2);

	gpuErrchk(cudaMemset(dev_hit, 0, sizeof(int) * nParticles * nSurfaces));

	return std::make_pair(dev_constraints, nCollisions);
	
}



