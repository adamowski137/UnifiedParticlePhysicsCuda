#pragma once

#include <utility>
#include <vector>
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
	void addStaticConstraints(T* constraints, int nConstraints);
	void setCpuConstraints(std::vector<T*> constraints);

	void addDynamicConstraints(T* constrains, int nConstrains);
	std::pair<T*, int> getConstraints();
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
	void reallocate(int nToFit);
	int nStaticConstraints;
	int nDynamicConstraints;
	int maxConstraints;
	bool initiated = false;

	T* storedConstraints;
	std::vector<T*> cpuConstraints;
};

template<typename T>
void ConstraintStorage<T>::addStaticConstraints(T* constraints, int nConstraints)
{
	int nSetConstraints = nStaticConstraints + nDynamicConstraints;
	if (nConstraints + nSetConstraints > maxConstraints)
		reallocate(nConstraints + nSetConstraints);
	gpuErrchk(cudaMemcpy(storedConstraints + nStaticConstraints, constraints, nConstraints * sizeof(T), cudaMemcpyHostToDevice));

	nStaticConstraints += nConstraints;
}

template<typename T>
void ConstraintStorage<T>::setCpuConstraints(std::vector<T*> constraints)
{
	cpuConstraints = constraints;
}

template<typename T>
void ConstraintStorage<T>::addDynamicConstraints(T* constraints, int nConstraints)
{
	if (maxConstraints < nDynamicConstraints + nStaticConstraints + nConstraints)
	{
		reallocate(nDynamicConstraints + nStaticConstraints + nConstraints);
	}
	gpuErrchk(cudaMemcpy(storedConstraints + nStaticConstraints + nDynamicConstraints, constraints, nConstraints * sizeof(T), cudaMemcpyDeviceToDevice));
	nDynamicConstraints += nConstraints;
}

template<typename T>
std::pair<T*, int> ConstraintStorage<T>::getConstraints()
{
	return std::make_pair(storedConstraints, nStaticConstraints + nDynamicConstraints);
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
	nDynamicConstraints = 0;
	if (!dynamic)
		nStaticConstraints = 0;
}

template<typename T>
void ConstraintStorage<T>::initInstance()
{
	if (!initiated)
	{
		gpuErrchk(cudaMalloc((void**)&storedConstraints, DEFAULT_CONSTRAINTS * sizeof(T)));
		maxConstraints = DEFAULT_CONSTRAINTS;
		nStaticConstraints = 0;
		nDynamicConstraints = 0;
		cpuConstraints.clear();
		initiated = true;
	}
}

template<typename T>
ConstraintStorage<T>::~ConstraintStorage()
{
	gpuErrchk(cudaFree(storedConstraints));
}

template<typename T>
void ConstraintStorage<T>::reallocate(int nToFit)
{
	T* new_constraints;
	while (maxConstraints < nToFit)
	{
		maxConstraints <<= 1;
	}

	gpuErrchk(cudaMalloc((void**)&new_constraints, sizeof(T) * maxConstraints));
	gpuErrchk(cudaMemcpy(new_constraints, storedConstraints, sizeof(T) * (nDynamicConstraints + nStaticConstraints), cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaFree(storedConstraints));
	storedConstraints = new_constraints;
}

template<typename T>
int ConstraintStorage<T>::getTotalConstraints()
{
	return nStaticConstraints + nDynamicConstraints;
}