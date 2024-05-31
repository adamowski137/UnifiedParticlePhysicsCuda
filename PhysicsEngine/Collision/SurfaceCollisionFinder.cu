#include "SurfaceCollisionFinder.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../GpuErrorHandling.hpp"

#include "thrust/device_ptr.h"
#include "thrust/device_vector.h"
#include "thrust/scan.h"
#include "../Constants.hpp"
#include "../Constraint/ConstraintStorage.cuh"
#include "../Config/Config.hpp"


__global__ void findSurfaceCollisions(
	int nParticles, int nSurfaces,
	int* hits, int* hit_sign, Surface* surfaces,
	float* x, float* y, float* z, float r)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= nParticles) return;

	for (int i = 0; i < nSurfaces; i++)
	{
		Surface s = surfaces[i];
		float v1 = s.a * x[index] + s.b * y[index] + s.c * z[index] + s.d;
		float sign = v1 < 0 ? 1 : -1;
		if (hits[i * nParticles + index] == 1)
		{
			if (hit_sign[i * nParticles + index] == -sign)
				hits[i * nParticles + index] = 0;
		}
		else
		{
			float v2 = s.a * (x[index] + sign * s.normal[0] * r) + s.b * (y[index] + sign * s.normal[1] * r) + s.c * (z[index] + sign * s.normal[2] * r) + s.d;

			if (v1 * v2 < 0) // hit
			{
				hits[i * nParticles + index] = 1;
				hit_sign[i * nParticles + index] = -sign;
			}
		}
	}
}

__global__ void fillConstraints(int nParticles, int nSurfaces,
	SurfaceConstraint* constraints,
	int* hits, int* hitsSum, Surface* surfaces, float r, float k)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= nParticles) return;

	for (int i = 0; i < nSurfaces; i++)
	{
		if (hits[index + i * nParticles])
		{
			// offset by 1 because we index array starting from 0
			int constraintIndex = hitsSum[index + i * nParticles] - 1;
			constraints[constraintIndex] = SurfaceConstraint().init(r, k, index, surfaces[i]);
		}
	}
}

SurfaceCollisionFinder::SurfaceCollisionFinder(std::vector<Surface> surfaces, int nParticles)
{
	nSurfaces = surfaces.size();
	gpuErrchk(cudaMalloc((void**)&dev_hit, sizeof(int) * nParticles * surfaces.size()));
	gpuErrchk(cudaMemset(dev_hit, 0, sizeof(int) * nParticles * surfaces.size()));

	gpuErrchk(cudaMalloc((void**)&dev_hit_sign, sizeof(int) * nParticles * surfaces.size()));
	gpuErrchk(cudaMemset(dev_hit_sign, 0, sizeof(int) * nParticles * surfaces.size()));

	gpuErrchk(cudaMalloc((void**)&dev_surface, sizeof(Surface) * surfaces.size()));
	gpuErrchk(cudaMemcpy(dev_surface, surfaces.data(), sizeof(Surface) * surfaces.size(), cudaMemcpyHostToDevice));

	nConstraintsMaxAllocated = 100;
	gpuErrchk(cudaMalloc((void**)&dev_foundCollisions, sizeof(SurfaceConstraint) * nConstraintsMaxAllocated));
}

SurfaceCollisionFinder::~SurfaceCollisionFinder()
{
	gpuErrchk(cudaFree(dev_hit));
	gpuErrchk(cudaFree(dev_hit_sign));
	gpuErrchk(cudaFree(dev_surface));
	gpuErrchk(cudaFree(dev_foundCollisions));
}

void SurfaceCollisionFinder::setSurfaces(std::vector<Surface> surfaces, int nParticles)
{
	gpuErrchk(cudaFree(dev_hit));
	gpuErrchk(cudaFree(dev_hit_sign));
	gpuErrchk(cudaFree(dev_surface));

	nSurfaces = surfaces.size();

	gpuErrchk(cudaMalloc((void**)&dev_hit, sizeof(int) * nParticles * surfaces.size()));
	gpuErrchk(cudaMemset(dev_hit, 0, sizeof(int) * nParticles * surfaces.size()));

	gpuErrchk(cudaMalloc((void**)&dev_hit_sign, sizeof(int) * nParticles * surfaces.size()));
	gpuErrchk(cudaMemset(dev_hit_sign, 0, sizeof(int) * nParticles * surfaces.size()));

	gpuErrchk(cudaMalloc((void**)&dev_surface, sizeof(Surface) * surfaces.size()));
	gpuErrchk(cudaMemcpy(dev_surface, surfaces.data(), sizeof(Surface) * surfaces.size(), cudaMemcpyHostToDevice));

}

void SurfaceCollisionFinder::findAndUpdateCollisions(int nParticles, float* x, float* y, float* z)
{
	int threads = 32;
	int blocks = (nParticles + threads - 1) / threads;

	findSurfaceCollisions << <blocks, threads >> > (nParticles, nSurfaces, dev_hit, dev_hit_sign, dev_surface, x, y, z, PARTICLERADIUS);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::device_vector<int> prefixSum(nParticles * nSurfaces);
	thrust::device_ptr<int> p = thrust::device_pointer_cast<int>(dev_hit);
	thrust::inclusive_scan(p, p + nSurfaces * nParticles, prefixSum.begin());

	int nCollisions = prefixSum[nParticles * nSurfaces - 1];
	if (nCollisions > nConstraintsMaxAllocated)
	{
		while (nCollisions > nConstraintsMaxAllocated)
			nConstraintsMaxAllocated *= 2;

		gpuErrchk(cudaFree(dev_foundCollisions));
		gpuErrchk(cudaMalloc((void**)&dev_foundCollisions, sizeof(SurfaceConstraint) * nConstraintsMaxAllocated));
	}


	int* dev_hitsSum = thrust::raw_pointer_cast(prefixSum.data());


	fillConstraints << <blocks, threads >> > (nParticles, nSurfaces,
		dev_foundCollisions,
		dev_hit, dev_hitsSum, dev_surface, PARTICLERADIUS, EngineConfig::K_SURFACE_CONSTRAINT);
	
	if(nCollisions > 0)
		ConstraintStorage<SurfaceConstraint>::Instance.addDynamicConstraints(dev_foundCollisions, nCollisions);

}



