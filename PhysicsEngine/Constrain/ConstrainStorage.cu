#include "ConstrainStorage.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void addCollisionsKern(List* collisions, int* counts, DistanceConstrain* constraints, ConstraintLimitType type, float d, int nParticles)
{
	const int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= nParticles - 1) return;
	Node* p = collisions[index].head;
	int constrainIndex = counts[index] - 1;

	while (p != NULL)
	{
		constraints[constrainIndex] = DistanceConstrain().init(d, index, p->value, type);
		p = p->next;
		constrainIndex--;
	}
}

__device__ __constant__ DistanceConstrain CUDAConstants::staticDistanceConstraints[MAX_CONSTRAINS];
__device__ __constant__ SurfaceConstraint CUDAConstants::staticSurfaceConstraints[MAX_CONSTRAINS];

ConstrainStorage ConstrainStorage::Instance;



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

	addCollisionsKern< <<particle_bound_blocks, threads> >>(collisions, counts, dynamicDistanceConstraints, type, d, nParticles);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}
