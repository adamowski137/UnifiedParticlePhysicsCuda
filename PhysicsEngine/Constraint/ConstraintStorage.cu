#include "ConstraintStorage.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

//__device__ __constant__ DistanceConstraint CUDAConstants::staticDistanceConstraints[MAX_CONSTRAINS];
//__device__ __constant__ SurfaceConstraint CUDAConstants::staticSurfaceConstraints[MAX_CONSTRAINS];

ConstraintStorage<DistanceConstraint> ConstraintStorage<DistanceConstraint>::Instance;
ConstraintStorage<SurfaceConstraint> ConstraintStorage<SurfaceConstraint>::Instance;
ConstraintStorage<RigidBodyConstraint> ConstraintStorage<RigidBodyConstraint>::Instance;

