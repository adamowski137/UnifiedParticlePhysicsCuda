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
	ConstraintStorage<RigidBodyConstraint>::Instance.initInstance();

	builder.args.dx = dev_dx;
	builder.args.dy = dev_dy;
	builder.args.dz = dev_dz;
}

ConstraintSolver::~ConstraintSolver()
{
	gpuErrchk(cudaFree(dev_dx));
	gpuErrchk(cudaFree(dev_dy));
	gpuErrchk(cudaFree(dev_dz));
}

void ConstraintSolver::initConstraintArgsBuilder(
	float* x, float* y, float* z,
	float* new_x, float* new_y, float* new_z,
	int* SDF_mode, float* SDF_value, float* SDF_normal_x, float* SDF_normal_y, float* SDF_normal_z,
	float* invmass)
{
	this->builder.initBase(new_x, new_y, new_z, 
		SDF_mode, SDF_value, SDF_normal_x, SDF_normal_y, SDF_normal_z,
		invmass);

	this->builder.addOldPosition(x, y, z);
}

void ConstraintSolver::clearAllConstraints()
{
	ConstraintStorage<DistanceConstraint>::Instance.clearConstraints(true);
	ConstraintStorage<SurfaceConstraint>::Instance.clearConstraints(true);
}
