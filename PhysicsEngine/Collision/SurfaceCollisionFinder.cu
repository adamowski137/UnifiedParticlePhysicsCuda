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
	int* hit, int* hit_sign, Surface* surfaces,
	float* x, float* y, float* z,
	float* new_x, float* new_y, float* new_z,
	float r)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= nParticles) return;

	for (int i = 0; i < nSurfaces; i++)
	{
		Surface s = surfaces[i];
		float v1 = s.a * new_x[index] + s.b * new_y[index] + s.c * new_z[index] + s.d;
		float sign_v1 = v1 < 0 ? 1 : -1;
		float direction = hit_sign[i * nParticles + index] == 0 ? sign_v1 : hit_sign[i * nParticles + index];

		float v1_off_r = s.a * (new_x[index] + direction * s.normal[0] * r) + s.b * (new_y[index] + direction * s.normal[1] * r) + s.c * (new_z[index] + direction * s.normal[2] * r) + s.d;
		float sign_v1_r = v1_off_r < 0 ? 1 : -1;

		if (hit[i * nParticles + index] == 1)
		{ 
			if (hit_sign[i * nParticles + index] == -sign_v1_r)
			{
				hit[i * nParticles + index] = 0;
				hit_sign[i * nParticles + index] = 0;
			}
		}
		else
		{
			float v2 = s.a * (new_x[index] + direction * s.normal[0] * r) + s.b * (new_y[index] + direction * s.normal[1] * r) + s.c * (new_z[index] + direction * s.normal[2] * r) + s.d;

			if (v1 * v2 < 0) // hit
			{
				hit[i * nParticles + index] = 1;
				hit_sign[i * nParticles + index] = sign_v1;
			}
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
			constraints[constraintIndex] = SurfaceConstraint().init(r, index, surfaces[i]);
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
}

SurfaceCollisionFinder::~SurfaceCollisionFinder()
{
	gpuErrchk(cudaFree(dev_hit));
	gpuErrchk(cudaFree(dev_hit_sign));
	gpuErrchk(cudaFree(dev_surface));
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

std::pair<SurfaceConstraint*, int> SurfaceCollisionFinder::findAndUpdateCollisions(int nParticles,
	float* x, float* y, float* z,
	float* new_x, float* new_y, float* new_z)
{
	if (nSurfaces == 0)
		return std::make_pair((SurfaceConstraint*)0, 0);

	int threads = 32;
	int blocks = (nParticles + threads - 1) / threads;

	findSurfaceCollisions << <blocks, threads >> > (nParticles, nSurfaces,
		dev_hit, dev_hit_sign, dev_surface,
		x, y, z,
		new_x, new_y, new_z,
		PARTICLERADIUS);

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
		dev_hit, dev_hitsSum, dev_surface, PARTICLERADIUS);

	//gpuErrchk(cudaMemset(dev_hit, 0, sizeof(int) * nParticles * nSurfaces));

	return std::make_pair(dev_constraints, nCollisions);

}



