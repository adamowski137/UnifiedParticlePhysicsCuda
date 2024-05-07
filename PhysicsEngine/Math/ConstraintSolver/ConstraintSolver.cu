#include "ConstraintSolver.cuh"
#include "../../Constraint/ConstraintStorage.cuh"

ConstraintSolver::ConstraintSolver(int particles) : nParticles{particles}
{
	gpuErrchk(cudaMalloc((void**)&dev_dx, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_dy, nParticles * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_dz, nParticles * sizeof(float)));

	gpuErrchk(cudaMemset(dev_dx, 0, nParticles * sizeof(float)));
	gpuErrchk(cudaMemset(dev_dy, 0, nParticles * sizeof(float)));
	gpuErrchk(cudaMemset(dev_dz, 0, nParticles * sizeof(float)));

	ConstraintStorage<DistanceConstraint>::Instance.initInstance();
	ConstraintStorage<SurfaceConstraint>::Instance.initInstance();
}

ConstraintSolver::~ConstraintSolver()
{
	gpuErrchk(cudaFree(dev_dx));
	gpuErrchk(cudaFree(dev_dy));
	gpuErrchk(cudaFree(dev_dz));
}

void ConstraintSolver::clearAllConstraints()
{
	ConstraintStorage<DistanceConstraint>::Instance.clearConstraints();
	ConstraintStorage<SurfaceConstraint>::Instance.clearConstraints();
}
