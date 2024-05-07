#pragma once

#include <utility>
#include <cuda_runtime.h>
#include "../List/List.cuh"
#include "thrust/device_ptr.h"
#include "../GpuErrorHandling.hpp"
#include "DistanceConstraint/DistanceConstraint.cuh"
#include "SurfaceConstraint/SurfaceConstraint.cuh"
#include "RigidBodyConstraint/RigidBodyConstraint.cuh"

#define MAX_CONSTRAINTS 512 
#define DEFAULT_CONSTRAINTS 64
// update when new constrain type added

//namespace CUDAConstants
//{
//	extern __device__ __constant__ DistanceConstraint staticDistanceConstraints[MAX_CONSTRAINTS];
//	extern __device__ __constant__ SurfaceConstraint staticSurfaceConstraints[MAX_CONSTRAINTS];
//}

template<typename T>
class ConstraintStorage 
{
public:
	void setStaticConstraints(T* constraints, int nConstraints);
	void setCpuConstraints(std::vector<T*> constraints);

	void addDynamicConstraints(T* constrains, int nConstrains);
	std::pair<T*, int> getStaticConstraints();
	std::pair<T*, int> getDynamicConstraints();
	std::pair<T*, int> getConstraints(bool dynamic);
	std::vector<T*> getCpuConstraints() { return cpuConstraints; }

	ConstraintStorage& operator=(const ConstraintStorage& other) = delete;
	ConstraintStorage(const ConstraintStorage& w) = delete;

	int getTotalConstraints();
	static ConstraintStorage<T> Instance;
	void clearConstraints();
	void clearConstraints(bool dynamic);
	void initInstance();
	~ConstraintStorage();
private:
	ConstraintStorage() {};
	int nStaticConstraints;
	int nDynamicConstraints;
	int maxDynamicConstraints;

	T* dynamicConstraints;
	T* staticConstraints;
	std::vector<T*> cpuConstraints;
};

template<typename T>
void ConstraintStorage<T>::setStaticConstraints(T* constraints, int nConstraints)
{
	if (staticConstraints != 0)
		gpuErrchk(cudaFree(staticConstraints));
	gpuErrchk(cudaMalloc((void**)&staticConstraints, nConstraints * sizeof(T)));
	gpuErrchk(cudaMemcpy(staticConstraints, constraints, nConstraints * sizeof(T), cudaMemcpyHostToDevice));

	nStaticConstraints = nConstraints;
}

template<typename T>
void ConstraintStorage<T>::setCpuConstraints(std::vector<T*> constraints)
{
	cpuConstraints = constraints;
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
	return std::pair<T*, int>(staticConstraints, nStaticConstraints);
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
	nStaticConstraints = 0;
	cpuConstraints.clear();
}

template<typename T>
void ConstraintStorage<T>::clearConstraints(bool dynamic)
{
	if (dynamic)
		nDynamicConstraints = 0;
	else
		nStaticConstraints = 0;
}

template<typename T>
void ConstraintStorage<T>::initInstance()
{
	gpuErrchk(cudaMalloc((void**)&dynamicConstraints, DEFAULT_CONSTRAINTS * sizeof(T)));

	staticConstraints = 0;

	nStaticConstraints = 0;
	nDynamicConstraints = 0;
	maxDynamicConstraints = DEFAULT_CONSTRAINTS;
	cpuConstraints.clear();
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