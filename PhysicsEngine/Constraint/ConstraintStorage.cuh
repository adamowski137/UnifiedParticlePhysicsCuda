#pragma once

#include <utility>
#include <cuda_runtime.h>
#include "../List/List.cuh"
#include "thrust/device_ptr.h"
#include "../GpuErrorHandling.hpp"
#include "DistanceConstraint/DistanceConstraint.cuh"
#include "SurfaceConstraint/SurfaceConstraint.cuh"

#define MAX_CONSTRAINS 128
#define DEFAULT_CONSTRAINS 64
// update when new constrain type added
#define CONSTRAINTYPESNUMBER 2

enum class ConstraintType
{
	DISTANCE,
	SURFACE
};

namespace CUDAConstants
{
	extern __device__ __constant__ DistanceConstraint staticDistanceConstraints[MAX_CONSTRAINS];
	extern __device__ __constant__ SurfaceConstraint staticSurfaceConstraints[MAX_CONSTRAINS];
}



class ConstraintStorage 
{
public:
	

	template<typename T>
	void setStaticConstraints(T* constrains, int nConstrains, ConstraintType type);
	template<typename T>
	void setDynamicConstraints(T* constrains, int nConstrains, ConstraintType type);	
	template<typename T>
	std::pair<T*, int> getStaticConstraints(ConstraintType type);
	template<typename T>
	std::pair<T*, int> getDynamicConstraints(ConstraintType type);
	template<typename T>
	std::pair<T*, int> getConstraints(ConstraintType type, bool dynamic);

	ConstraintStorage& operator=(const ConstraintStorage& other) = delete;
	ConstraintStorage(const ConstraintStorage& w) = delete;

	int getTotalConstraints();
	void addCollisions(List* collisions, int* sums, ConstraintLimitType type, float d, int nParticles);
	static ConstraintStorage Instance;
	void clearConstraints();
	void initInstance();
	~ConstraintStorage();
private:
	ConstraintStorage() {};
	int nStaticConstraints[CONSTRAINTYPESNUMBER];
	int nDynamicConstraints[CONSTRAINTYPESNUMBER];
	int maxDynamicConstraints[CONSTRAINTYPESNUMBER];
	int maxConstraints[CONSTRAINTYPESNUMBER];

	DistanceConstraint* dynamicDistanceConstraints;
	SurfaceConstraint* dynamicSurfaceConstraints;
};

template<typename T>
void ConstraintStorage::setStaticConstraints(T* constrains, int nConstrains, ConstraintType type)
{
	nStaticConstraints[(int)type] = nConstrains;
	if (type == ConstraintType::DISTANCE)
	{
		gpuErrchk(cudaMemcpyToSymbol(CUDAConstants::staticDistanceConstraints, constrains, nConstrains * sizeof(T)));
	}
	if (type == ConstraintType::SURFACE)
	{
		gpuErrchk(cudaMemcpyToSymbol(CUDAConstants::staticSurfaceConstraints, constrains, nConstrains * sizeof(T)));
	}
}

template<typename T>
void ConstraintStorage::setDynamicConstraints(T* constraints, int nConstraints, ConstraintType type)
{
	nDynamicConstraints[(int)type] = nConstraints;
	bool reallocate = false;
	if (maxDynamicConstraints[(int)type] < nConstraints)
	{
		maxDynamicConstraints[(int)type] = nConstraints;
		reallocate = true;
	}

	if (type == ConstraintType::DISTANCE)
	{
		if (reallocate)
		{
			gpuErrchk(cudaFree(dynamicDistanceConstraints));
			gpuErrchk(cudaMalloc((void**)&dynamicDistanceConstraints, nConstraints * sizeof(T)));
		}
		gpuErrchk(cudaMemcpy(dynamicDistanceConstraints, constraints, nConstraints * sizeof(T), cudaMemcpyDeviceToDevice));
		
	}
	if (type == ConstraintType::SURFACE)
	{
		if (reallocate)
		{
			gpuErrchk(cudaFree(dynamicSurfaceConstraints));
			gpuErrchk(cudaMalloc((void**)&dynamicSurfaceConstraints, nConstraints * sizeof(T)));
		}
		gpuErrchk(cudaMemcpy(dynamicSurfaceConstraints, constraints, nConstraints * sizeof(T), cudaMemcpyDeviceToDevice));
	}
}

template<typename T>
std::pair<T*, int> ConstraintStorage::getStaticConstraints(ConstraintType type)
{
	if (type == ConstraintType::DISTANCE)
		return std::pair<T*, int>((T*)CUDAConstants::staticDistanceConstraints, nStaticConstraints[(int)type]);
	if (type == ConstraintType::SURFACE)
		return std::pair<T*, int>((T*)CUDAConstants::staticSurfaceConstraints, nStaticConstraints[(int)type]);
}

template<typename T>
std::pair<T*, int> ConstraintStorage::getDynamicConstraints(ConstraintType type)
{
	if(type == ConstraintType::DISTANCE)
		return std::pair<T*, int>((T*)dynamicDistanceConstraints, nDynamicConstraints[(int)type]);
	if (type == ConstraintType::SURFACE)
		return std::pair<T*, int>((T*)dynamicSurfaceConstraints, nDynamicConstraints[(int)type]);
}

template<typename T>
std::pair<T*, int> ConstraintStorage::getConstraints(ConstraintType type, bool dynamic)
{
	if(dynamic)
		return getDynamicConstraints<T>(type);
	else
		return getStaticConstraints<T>(type);
}

