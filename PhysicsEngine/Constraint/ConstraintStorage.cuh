#pragma once

#include <utility>
#include <cuda_runtime.h>
#include "../List/List.cuh"
#include "thrust/device_ptr.h"
#include "../GpuErrorHandling.hpp"
#include "DistanceConstraint/DistanceConstraint.cuh"
#include "SurfaceConstraint/SurfaceConstraint.cuh"
#include "RigidBodyConstraint/RigidBodyConstraint.cuh"

#define MAX_CONSTRAINS 512 
#define DEFAULT_CONSTRAINS 64
// update when new constrain type added

namespace CUDAConstants
{
	extern __device__ __constant__ DistanceConstraint staticDistanceConstraints[MAX_CONSTRAINS];
	extern __device__ __constant__ SurfaceConstraint staticSurfaceConstraints[MAX_CONSTRAINS];
}

template<typename T>
class ConstraintStorage 
{
public:
	void addStaticConstraints(std::shared_ptr<T> constraints, int nConstraints);
	void addDynamicConstraints(T* constrains, int nConstrains);
	std::pair<T*, int> getStaticConstraints();
	std::pair<T*, int> getDynamicConstraints();
	std::pair<T*, int> getConstraints(bool dynamic);

	ConstraintStorage& operator=(const ConstraintStorage& other) = delete;
	ConstraintStorage(const ConstraintStorage& w) = delete;

	int getTotalConstraints();
	static ConstraintStorage<T> Instance;
	void clearConstraints();
	void initInstance();
	~ConstraintStorage();
private:
	ConstraintStorage() {};
	int nStaticConstraints;
	int nDynamicConstraints;
	int maxDynamicConstraints;
	int maxConstraints;

	T* dynamicConstraints;
	std::shared_ptr<T> staticConstraints;
};

template<typename T>
void ConstraintStorage<T>::addStaticConstraints(std::shared_ptr<T> constraints, int nConstraints)
{
	nStaticConstraints = nConstraints;
	staticConstraints = constraints;
}

template<typename T>
void ConstraintStorage<T>::addDynamicConstraints(T* constraints, int nConstraints)
{
	if (maxDynamicConstraints < nDynamicConstraints + nConstraints)
	{
		int targetCapacity = nDynamicConstraints + nConstraints;
		while (maxDynamicConstraints < targetCapacity)
			maxDynamicConstraints <<= 1;

		gpuErrchk(cudaFree(dynamicConstraints));
		gpuErrchk(cudaMalloc((void**)&dynamicConstraints, maxDynamicConstraints * sizeof(T)));
	}
	gpuErrchk(cudaMemcpy(dynamicConstraints + nDynamicConstraints, constraints, nConstraints * sizeof(T), cudaMemcpyDeviceToDevice));
	nDynamicConstraints += nConstraints;
}

template<typename T>
std::pair<T*, int> ConstraintStorage<T>::getStaticConstraints()
{
	return std::pair<T*, int>(staticConstraints.get(), nStaticConstraints);
}

template<typename T>
std::pair<T*, int> ConstraintStorage<T>::getDynamicConstraints()
{
	return std::pair<T*, int>(dynamicConstraints, nDynamicConstraints);
}

template<typename T>
std::pair<T*, int> ConstraintStorage<T>::getConstraints(bool dynamic)
{
	if(dynamic)
		return getDynamicConstraints();
	else
		return getStaticConstraints();
}

template<typename T>
void ConstraintStorage<T>::clearConstraints()
{
	nDynamicConstraints = 0;
}

template<typename T>
void ConstraintStorage<T>::initInstance()
{
	gpuErrchk(cudaMalloc((void**)&dynamicConstraints, DEFAULT_CONSTRAINS * sizeof(T)));

	nStaticConstraints = 0;
	nDynamicConstraints = 0;
	maxDynamicConstraints = DEFAULT_CONSTRAINS;
}

template<typename T>
ConstraintStorage<T>::~ConstraintStorage()
{
	gpuErrchk(cudaFree(dynamicConstraints));
}

template<typename T>
int ConstraintStorage<T>::getTotalConstraints()
{
	return nStaticConstraints + nDynamicConstraints;
}