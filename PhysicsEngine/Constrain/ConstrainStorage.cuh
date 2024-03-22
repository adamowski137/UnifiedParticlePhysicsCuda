#pragma once
#include <cuda_runtime.h>
#include "Constrain/Constrain.cuh"
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

__device__ __constant__ Constrain staticConstrains[CONSTRAINTYPESNUMBER][MAX_CONSTRAINS];


class ConstrainStorage 
{
public:
	template<typename T>
	void setStaticConstrains(T* constrains, int nConstrains, ConstrainType type);
	template<typename T>
	void setDynamicConstrains(T* constrains, int nConstrains, ConstrainType type);
	template<typename T>
	std::pair<T*, int> getStaticConstrains(ConstrainType type);
	template<typename T>
	std::pair<T*, int> getDynamicConstrains(ConstrainType type);

	static ConstrainStorage getInstance();
private:
	ConstrainStorage();

	int allocatedDynamicConstrains[CONSTRAINTYPESNUMBER];
	int nStaticConstrains[CONSTRAINTYPESNUMBER];
	int nDynamicConstrains[CONSTRAINTYPESNUMBER];
	Constrain** dynamicConstrains;
};

ConstrainStorage::ConstrainStorage()
{
	for (int i = 0; i < CONSTRAINTYPESNUMBER; i++)
	{
		allocatedDynamicConstrains[i] = 
	}
}


template<typename T>
void ConstrainStorage::setStaticConstrains(T* constrains, int nConstrains, ConstrainType type)
{
	gpuErrchk(cudaMemcpyToSymbol(staticConstrains[type], constrains, nConstrains * sizeof(T));
	nStaticConstrains[type] = nConstrains;
}

template<typename T>
void ConstrainStorage::setDynamicConstrains(T* constrains, int nConstrains, ConstrainType type)
{
		gpuErrchk(cudaMemcpy(dynamicConstrains[type], constrains, nConstrains * sizeof(T), cudaMemCpyHostToDevice));
		nDynamicConstrains[type] = nConstrains;
}

template<typename T>
std::pair<T*, int> ConstrainStorage::getStaticConstrains(ConstrainType type)
{
	return std::pair<T*, int>((T*)staticConstrains[type], nStaticDistanceConstrains);
}

template<typename T>
std::pair<T*, int> ConstrainStorage::getDynamicConstrains(ConstrainType type)
{
	return std::pair<T, int>((T*)dynamicConstrains[type], nDynamicConstrains);
}

ConstrainStorage ConstrainStorage::getInstance()
{
	return ConstrainStorage();
}
