#pragma once

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "DistanceConstrain/DistanceConstrain.cuh"
#include "SurfaceConstraint/SurfaceConstraint.cuh"
#include "../GpuErrorHandling.hpp"
#include <utility>
#define MAX_CONSTRAINS 128
#define DEFAULT_CONSTRAINS 64
// update when new constrain type added
#define CONSTRAINTYPESNUMBER 2

enum class ConstrainType
{
	DISTANCE,
	SURFACE
};

__device__ __constant__ DistanceConstrain staticDistanceConstraints[MAX_CONSTRAINS];
__device__  __constant__ SurfaceConstraint staticSurfaceConstraints[MAX_CONSTRAINS];


class ConstrainStorage 
{
public:
	template<typename T>
	void setStaticConstraints(T* constrains, int nConstrains, ConstrainType type);
	template<typename T>
	void setDynamicConstraints(T* constrains, int nConstrains, ConstrainType type);	
	template<typename T>
	std::pair<T*, int> getStaticConstraints(ConstrainType type);
	template<typename T>
	std::pair<T*, int> getDynamicConstraints(ConstrainType type);
	template<typename T>
	std::pair<T*, int> getConstraints(ConstrainType type);

	ConstrainStorage& operator=(const ConstrainStorage& other) = delete;
	ConstrainStorage(const ConstrainStorage& w) = delete;


	void addCollisions(List* collisions, int* counts, ConstraintLimitType type, float d, int nParticles);
	static ConstrainStorage Instance;
	void initInstance();
	~ConstrainStorage();
private:
	ConstrainStorage();
	int nStaticConstraints[CONSTRAINTYPESNUMBER];
	int nDynamicConstraints[CONSTRAINTYPESNUMBER];
	int maxDynamicConstraints[CONSTRAINTYPESNUMBER];
	int maxConstraints[CONSTRAINTYPESNUMBER];

	DistanceConstrain* dynamicDistanceConstraints;
	SurfaceConstraint* dynamicSurfaceConstraints;

	DistanceConstrain* distanceConstraints;
	SurfaceConstraint* surfaceConstraints;

};

//__global__ void addCollisionsKern(List* collisions, int* counts, DistanceConstrain* constraints, ConstraintLimitType type, float d, int nParticles)
//{
//	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
//	if (index >= nParticles - 1) return;
//	Node* p = collisions[index].head;
//	int constrainIndex = counts[index] - 1;
//
//	while (p != NULL)
//	{
//		constraints[constrainIndex] = DistanceConstrain().init(d, index, p->value, type);
//		p = p->next;
//		constrainIndex--;
//	}
//}

void ConstrainStorage::initInstance()
{
	gpuErrchk(cudaMalloc((void**)&dynamicDistanceConstraints, DEFAULT_CONSTRAINS * sizeof(DistanceConstrain)));
	gpuErrchk(cudaMalloc((void**)&dynamicSurfaceConstraints, DEFAULT_CONSTRAINS * sizeof(DistanceConstrain)));
	
	for (int i = 0; i < CONSTRAINTYPESNUMBER; i++)
	{
		nStaticConstraints[i] = 0;
		nDynamicConstraints[i] = 0;
		maxDynamicConstraints[i] = DEFAULT_CONSTRAINS;
	}
}

ConstrainStorage::~ConstrainStorage()
{
	gpuErrchk(cudaFree(dynamicDistanceConstraints));
	gpuErrchk(cudaFree(dynamicSurfaceConstraints));
}


template<typename T>
void ConstrainStorage::setStaticConstraints(T* constrains, int nConstrains, ConstrainType type)
{
	nStaticConstraints[(int)type] = nConstrains;
	if (type == ConstrainType::DISTANCE)
	{
		gpuErrchk(cudaMemcpyToSymbol(staticDistanceConstraints, constrains, nConstrains * sizeof(T)));
	}
	if (type == ConstrainType::SURFACE)
	{
		gpuErrchk(cudaMemcpyToSymbol(staticSurfaceConstraints, constrains, nConstrains * sizeof(T)));
	}
}

template<typename T>
void ConstrainStorage::setDynamicConstraints(T* constrains, int nConstrains, ConstrainType type)
{
	nDynamicConstraints[(int)type] = nConstrains;
	bool reallocate = false;
	if (maxDynamicConstraints[(int)type] < nConstrains)
	{
		maxDynamicConstraints[(int)type] = nConstrains;
		reallocate = true;
	}

	if (type == ConstrainType::DISTANCE)
	{
		if (reallocate)
		{
			gpuErrchk(cudaFree(dynamicDistanceConstraints));
			gpuErrchk(cudaMalloc((void**)&dynamicDistanceConstraints, nConstrains * sizeof(T)));
		}
		gpuErrchk(cudaMemcpy(dynamicDistanceConstraints, constrains, nConstrains * sizeof(T)));
		
	}
	if (type == ConstrainType::SURFACE)
	{
		if (reallocate)
		{
			gpuErrchk(cudaFree(dynamicSurfaceConstraints));
			gpuErrchk(cudaMalloc((void**)&dynamicSurfaceConstraints, nConstrains * sizeof(T)));
		}
		gpuErrchk(cudaMemcpy(dynamicSurfaceConstraints, constrains, nConstrains * sizeof(T)));
	}
}

template<typename T>
std::pair<T*, int> ConstrainStorage::getStaticConstraints(ConstrainType type)
{
	if (type == ConstrainType::DISTANCE)
		return std::pair<T*, int>(staticDistanceConstraints, nStaticConstraints[(int)type]);
	if (type == ConstrainType::SURFACE)
		return std::pair<T*, int>(staticSurfaceConstraints, nStaticConstraints[(int)type]);
}

template<typename T>
std::pair<T*, int> ConstrainStorage::getDynamicConstraints(ConstrainType type)
{
	if(type == ConstrainType::DISTANCE)
		return std::pair<T*, int>(dynamicDistanceConstraints, nDynamicConstraints[(int)type]);
	if (type == ConstrainType::SURFACE)
		return std::pair<T*, int>(dynamicSurfaceConstraints, nDynamicConstraints[(int)type]);
}

template<typename T>
std::pair<T*, int> ConstrainStorage::getConstraints(ConstrainType type)
{
	int n = nStaticConstraints[(int)type] + nDynamicConstraints[(int)type];
	bool reallocate = false;
	if(maxConstraints[(int)type] < n)
	{
		reallocate = true;
		maxConstraints[(int)type] = n;
	}
	if (type == ConstrainType::DISTANCE)
	{
		if (reallocate)
		{
			gpuErrchk(cudaFree(distanceConstraints));
			gpuErrchk(cudaMalloc((void**)&distanceConstraints, n * sizeof(T)));
		}
		gpuErrchk(cudaMemcpyFromSymbol(distanceConstraints, staticDistanceConstraints, nStaticConstraints[(int)type] * sizeof(T), 0, cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(distanceConstraints + nStaticConstraints[(int)type], dynamicDistanceConstraints, nDynamicConstraints[(int)type] * sizeof(T), cudaMemcpyDeviceToDevice));
		return std::pair<T*, int>((T*)distanceConstraints, nDynamicConstraints[(int)type] + nStaticConstraints[(int)type]);

	}
	if (type == ConstrainType::SURFACE)
	{
		if (reallocate)
		{
			gpuErrchk(cudaFree(surfaceConstraints));
			gpuErrchk(cudaMalloc((void**)&surfaceConstraints, n * sizeof(T)));
		}
		gpuErrchk(cudaMemcpyFromSymbol(surfaceConstraints, staticSurfaceConstraints, nStaticConstraints[(int)type] * sizeof(T)));
		gpuErrchk(cudaMemcpy(surfaceConstraints + nStaticConstraints[(int)type], dynamicSurfaceConstraints, nDynamicConstraints[(int)type] * sizeof(T), cudaMemcpyDeviceToDevice));
		return std::pair<T*, int>((T*)surfaceConstraints, nDynamicConstraints[(int)type] + nStaticConstraints[(int)type]);
	}

}

void ConstrainStorage::addCollisions(List* collisions, int* counts, ConstraintLimitType type, float d, int nParticles)
{
	nDynamicConstraints[(int)ConstrainType::DISTANCE] = counts[nParticles - 1];
	if (maxDynamicConstraints[(int)ConstrainType::DISTANCE] < counts[nParticles - 1])
	{
		maxDynamicConstraints[(int)ConstrainType::DISTANCE] = counts[nParticles - 1];
		gpuErrchk(cudaFree(dynamicDistanceConstraints));
		gpuErrchk(cudaMalloc((void**)&dynamicDistanceConstraints, counts[nParticles - 1] * sizeof(DistanceConstrain)));
	}

	unsigned int threads = 32;
	int particle_bound_blocks = (nParticles + threads - 1) / threads;

	//addCollisionsKern<< <particle_bound_blocks, threads> >>(collisions, counts, dynamicDistanceConstraints, type, d, nParticles);
	//gpuErrchk(cudaGetLastError());
	//gpuErrchk(cudaDeviceSynchronize());
}
