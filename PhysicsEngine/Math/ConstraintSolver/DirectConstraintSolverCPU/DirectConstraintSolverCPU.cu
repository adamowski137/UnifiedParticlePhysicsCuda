#include "DirectConstraintSolverCPU.cuh"
#include "../../../Constraint/ConstraintStorage.cuh"
#include "../../../GpuErrorHandling.hpp"
#include "thrust/device_ptr.h"

DirectConstraintSolverCPU::DirectConstraintSolverCPU(int nParticles) : ConstraintSolver(nParticles)
{
	x_cpu = new float[nParticles];
	y_cpu = new float[nParticles];
	z_cpu = new float[nParticles];
	invmass_cpu = new float[nParticles];
}

DirectConstraintSolverCPU::~DirectConstraintSolverCPU()
{
	delete[] x_cpu;
	delete[] y_cpu;
	delete[] z_cpu;
	delete[] invmass_cpu;
}

void DirectConstraintSolverCPU::calculateForces(float* new_x, float* new_y, float* new_z, float* invmass, int* dev_phase, float dt, int iterations)
{
	gpuErrchk(cudaMemcpy(x_cpu, new_x, sizeof(float) * nParticles, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(y_cpu, new_y, sizeof(float) * nParticles, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(z_cpu, new_z, sizeof(float) * nParticles, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(invmass_cpu, invmass, sizeof(float) * nParticles, cudaMemcpyDeviceToHost));

	for (int i = 0; i < iterations; i++)
	{
		this->projectConstraints<DistanceConstraint>(new_x, new_y, new_z, invmass, dev_phase, dt / iterations, iterations);
		this->projectConstraints<SurfaceConstraint>(new_x, new_y, new_z, invmass, dev_phase, dt / iterations, iterations);
	}

	gpuErrchk(cudaMemcpy(new_x, x_cpu, sizeof(float) * nParticles, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(new_y, y_cpu, sizeof(float) * nParticles, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(new_z, z_cpu, sizeof(float) * nParticles, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(invmass, invmass_cpu, sizeof(float) * nParticles, cudaMemcpyHostToDevice));

	clearAllConstraints();
}

void DirectConstraintSolverCPU::calculateStabilisationForces(float* x, float* y, float* z, int* mode, float* new_x, float* new_y, float* new_z, float* invmass, float dt, int iterations)
{
	throw - 1;
}

template<typename T>
void DirectConstraintSolverCPU::projectConstraints(float* x, float* y, float* z, float* invmass, int* phase, float dt, int iterations)
{

	auto staticConstraintData = ConstraintStorage<T>::Instance.getConstraints(false);
	auto dynamicConstraintData = ConstraintStorage<T>::Instance.getConstraints(true);
	int n = staticConstraintData.second + dynamicConstraintData.second;
	T* cpu_constraints = new T[n];
	//std::cout << n << "\n";

	gpuErrchk(cudaMemcpy(cpu_constraints, staticConstraintData.first, sizeof(T) * staticConstraintData.second, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(cpu_constraints + staticConstraintData.second, dynamicConstraintData.first, sizeof(T) * dynamicConstraintData.second, cudaMemcpyDeviceToHost));

	for (int i = 0; i < n; i++)
	{
		cpu_constraints[i].directSolve_cpu(x_cpu, y_cpu, z_cpu, invmass_cpu, dt);
	}

	delete[] cpu_constraints;
}
