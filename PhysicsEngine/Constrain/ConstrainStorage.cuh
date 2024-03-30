#pragma once

#include <cuda_runtime.h>
#include "DistanceConstrain/DistanceConstrain.cuh"
#include "SurfaceConstraint/SurfaceConstraint.cuh"
#include "../GpuErrorHandling.hpp"
#include <utility>
#include "../List/List.cuh"
#include "thrust/device_ptr.h"

#define MAX_CONSTRAINS 128
#define DEFAULT_CONSTRAINS 64
// update when new constrain type added
#define CONSTRAINTYPESNUMBER 2

enum class ConstrainType
{
	DISTANCE,
	SURFACE
};

namespace CUDAConstants
{
	extern __device__ __constant__ DistanceConstrain staticDistanceConstraints[MAX_CONSTRAINS];
	extern __device__ __constant__ SurfaceConstraint staticSurfaceConstraints[MAX_CONSTRAINS];
}



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
	std::pair<T*, int> getConstraints(ConstrainType type, bool dynamic);

	ConstrainStorage& operator=(const ConstrainStorage& other) = delete;
	ConstrainStorage(const ConstrainStorage& w) = delete;

	int getTotalConstraints();
	void addCollisions(List* collisions, int* sums, ConstraintLimitType type, float d, int nParticles);
	static ConstrainStorage Instance;
	void clearConstraints();
	void initInstance();
	~ConstrainStorage();
private:
	ConstrainStorage() {};
	int nStaticConstraints[CONSTRAINTYPESNUMBER];
	int nDynamicConstraints[CONSTRAINTYPESNUMBER];
	int maxDynamicConstraints[CONSTRAINTYPESNUMBER];
	int maxConstraints[CONSTRAINTYPESNUMBER];

	DistanceConstrain* dynamicDistanceConstraints;
	SurfaceConstraint* dynamicSurfaceConstraints;
};

template<typename T>
void ConstrainStorage::setStaticConstraints(T* constrains, int nConstrains, ConstrainType type)
{
	nStaticConstraints[(int)type] = nConstrains;
	if (type == ConstrainType::DISTANCE)
	{
		gpuErrchk(cudaMemcpyToSymbol(CUDAConstants::staticDistanceConstraints, constrains, nConstrains * sizeof(T)));
	}
	if (type == ConstrainType::SURFACE)
	{
		gpuErrchk(cudaMemcpyToSymbol(CUDAConstants::staticSurfaceConstraints, constrains, nConstrains * sizeof(T)));
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
		gpuErrchk(cudaMemcpy(dynamicDistanceConstraints, constrains, nConstrains * sizeof(T), cudaMemcpyDeviceToDevice));
		
	}
	if (type == ConstrainType::SURFACE)
	{
		if (reallocate)
		{
			gpuErrchk(cudaFree(dynamicSurfaceConstraints));
			gpuErrchk(cudaMalloc((void**)&dynamicSurfaceConstraints, nConstrains * sizeof(T)));
		}
		gpuErrchk(cudaMemcpy(dynamicSurfaceConstraints, constrains, nConstrains * sizeof(T), cudaMemcpyDeviceToDevice));
	}
}

template<typename T>
std::pair<T*, int> ConstrainStorage::getStaticConstraints(ConstrainType type)
{
	if (type == ConstrainType::DISTANCE)
		return std::pair<T*, int>((T*)CUDAConstants::staticDistanceConstraints, nStaticConstraints[(int)type]);
	if (type == ConstrainType::SURFACE)
		return std::pair<T*, int>((T*)CUDAConstants::staticSurfaceConstraints, nStaticConstraints[(int)type]);
}

template<typename T>
std::pair<T*, int> ConstrainStorage::getDynamicConstraints(ConstrainType type)
{
	if(type == ConstrainType::DISTANCE)
		return std::pair<T*, int>((T*)dynamicDistanceConstraints, nDynamicConstraints[(int)type]);
	if (type == ConstrainType::SURFACE)
		return std::pair<T*, int>((T*)dynamicSurfaceConstraints, nDynamicConstraints[(int)type]);
}

template<typename T>
std::pair<T*, int> ConstrainStorage::getConstraints(ConstrainType type, bool dynamic)
{
	if(dynamic)
		return getDynamicConstraints<T>(type);
	else
		return getStaticConstraints<T>(type);
}

