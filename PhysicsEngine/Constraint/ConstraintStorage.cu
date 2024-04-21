#include "ConstraintStorage.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

__global__ void addCollisionsKern(List* collisions, int* counts, DistanceConstraint* constraints, ConstraintLimitType type, float d, int nParticles)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles) return;
	Node* p = collisions[index].head;
	int constrainIndex = counts[index] - 1;
	while (p != NULL)
	{
		constraints[constrainIndex] = DistanceConstraint().init(d, index, p->value, type);
		p = p->next;
		constrainIndex--;
	}
}

__device__ __constant__ DistanceConstraint CUDAConstants::staticDistanceConstraints[MAX_CONSTRAINS];
__device__ __constant__ SurfaceConstraint CUDAConstants::staticSurfaceConstraints[MAX_CONSTRAINS];

ConstraintStorage ConstraintStorage::Instance;

void ConstraintStorage::clearConstraints()
{
	memset(nDynamicConstraints, 0, sizeof(int) * CONSTRAINTYPESNUMBER);
}

void ConstraintStorage::initInstance()
{
	gpuErrchk(cudaMalloc((void**)&dynamicDistanceConstraints, DEFAULT_CONSTRAINS * sizeof(DistanceConstraint)));
	gpuErrchk(cudaMalloc((void**)&dynamicSurfaceConstraints, DEFAULT_CONSTRAINS * sizeof(SurfaceConstraint)));
	
	for (int i = 0; i < CONSTRAINTYPESNUMBER; i++)
	{
		nStaticConstraints[i] = 0;
		nDynamicConstraints[i] = 0;
		maxDynamicConstraints[i] = DEFAULT_CONSTRAINS;
	}
}

ConstraintStorage::~ConstraintStorage()
{
	gpuErrchk(cudaFree(dynamicDistanceConstraints));
	gpuErrchk(cudaFree(dynamicSurfaceConstraints));
}


int ConstraintStorage::getTotalConstraints()
{
	int sum = 0;
	for(int i = 0; i < CONSTRAINTYPESNUMBER; i++)
	{
		sum += nStaticConstraints[i] + nDynamicConstraints[i];
	}
	return sum;
}

void ConstraintStorage::addCollisions(List* collisions, int* sums, ConstraintLimitType ctype, float d, int nParticles)
{
	thrust::device_ptr<int> counts(sums);
	int nCollisions = counts[nParticles - 1];

	if (nCollisions == 0) return;

	nDynamicConstraints[(int)ConstraintType::DISTANCE] = nCollisions;
	if (maxDynamicConstraints[(int)ConstraintType::DISTANCE] < nCollisions)
	{
		maxDynamicConstraints[(int)ConstraintType::DISTANCE] = nCollisions;
		gpuErrchk(cudaFree(dynamicDistanceConstraints));
		gpuErrchk(cudaMalloc((void**)&dynamicDistanceConstraints, nCollisions * sizeof(DistanceConstraint)));
	}

	int threads = 32;
	int particle_bound_blocks = (nParticles + threads - 1) / threads;

	addCollisionsKern<<<particle_bound_blocks, threads>>> (collisions, sums, dynamicDistanceConstraints, ctype, d, nParticles);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}
