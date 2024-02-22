#pragma once
#define THREADS 512

class ParticleType
{
public:
	ParticleType(int amount, float mass);
	~ParticleType();

	void renderData();
private:
	const int amountOfParticles;
	float* dev_x;
	float* dev_y;
	float* dev_z;
	float* dev_vx;
	float* dev_vy;
	float* dev_vz;
	float* dev_invmass;
	
	float* dev_invM;

	unsigned int vao;
	unsigned int vboSphere;
	unsigned int vbox;
	unsigned int vboy;
	unsigned int vboz;

	int blocks;

	void setupDeviceData();
	void setupShaderData();
};