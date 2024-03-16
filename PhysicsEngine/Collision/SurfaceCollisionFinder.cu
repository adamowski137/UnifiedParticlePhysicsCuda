#include "SurfaceCollisionFinder.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../GpuErrorHandling.hpp"

#include "thrust/device_ptr.h"
#include "thrust/device_vector.h"
#include "thrust/scan.h"

__host__ __device__ float calculateC(float x, float y, float z, float r, Surface s)
{
	float len = sqrtf(s.a * s.a + s.b * s.b + s.c * s.c);
	float dist = (s.a * x + s.b * y + s.c * z + s.d) / len;
	return dist - r;
}

__host__ __device__ float calculateTimeDerivative(float vx, float vy, float vz, Surface s)
{
	return s.normal[0] * vx + s.normal[1] * vy + s.normal[2] * vz;

}

__host__ __device__ void calculatePositionDerivative(Surface s, float* output)
{
	output[0] = s.normal[0];
	output[1] = s.normal[1];
	output[2] = s.normal[2];
}

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
		float sign = v1 < 0 ? -1 : 1;

		float v2 = s.a * (x[index] + sign * s.normal[0] * r) + s.b * (y[index] + sign * s.normal[1] * r) + s.c * (z[index] + sign * s.normal[2] * r) + s.d;

		if (v1 * v2 < 0) // hit
		{
			hits[i * nParticles + index] = 1;
		}
	}
}

__global__ void fillConstraints(int nParticles, int nSurfaces,
	float* constraints, float* resultVector,
	int* hits, int* hitsSum, Surface* surfaces,
	float* x, float* y, float* z, float* vx, float* vy, float* vz,
	float r)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= nParticles) return;

	for (int i = 0; i < nSurfaces; i++)
	{
		if (hits[index + i * nParticles])
		{
			int constraintIndex = hitsSum[index + i * nParticles];
			calculatePositionDerivative(surfaces[i], &constraints[constraintIndex * 3 * nParticles + index * 3]);
			resultVector[constraintIndex] = -calculateC(x[index], y[index], z[index], r, surfaces[i])
				- calculateTimeDerivative(vx[index], vy[index], vz[index], surfaces[i]);
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

std::pair<float*, float*> SurfaceCollisionFinder::findAndUpdateCollisions(int nParticles, float* x, float* y, float* z, float* vx, float* vy, float* vz)
{
	int threads = 32;
	int blocks = (nParticles + threads - 1) / threads;

	findSurfaceCollisions << <blocks, threads >> > (nParticles, nSurfaces, dev_hit, dev_surface, x, y, z, 2.f);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::device_vector<int> prefixSum(nParticles * nSurfaces);
	thrust::device_ptr<int> p = thrust::device_pointer_cast<int>(dev_hit);
	thrust::inclusive_scan(p, p + nSurfaces * nParticles, prefixSum.begin());

	int nCollisions = prefixSum[nParticles * nSurfaces - 1];
	float* dev_jacobian_surface_collisions, *dev_b_surface_collisions;

	gpuErrchk(cudaMalloc((void**)dev_jacobian_surface_collisions, sizeof(float) * 3 * nParticles * nCollisions));
	gpuErrchk(cudaMemset(dev_jacobian_surface_collisions, 0, sizeof(float) * 3 * nParticles * nCollisions));

	gpuErrchk(cudaMalloc((void**)dev_b_surface_collisions, sizeof(float) * nCollisions));
	gpuErrchk(cudaMemset(dev_b_surface_collisions, 0, sizeof(float) * nCollisions));

	int* dev_hitsSum = thrust::raw_pointer_cast(prefixSum.data());


	fillConstraints << <blocks, threads >> > (nParticles, nSurfaces,
		dev_jacobian_surface_collisions, dev_b_surface_collisions,
		dev_hit, dev_hitsSum, dev_surface, 
		x, y, z, vx, vy, vz, 2.f);

	return std::make_pair(dev_jacobian_surface_collisions, dev_b_surface_collisions);
	
}



