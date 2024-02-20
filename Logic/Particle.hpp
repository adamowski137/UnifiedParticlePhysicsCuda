class ParticleType
{
public:
	ParticleType(int amount, float mass);
	~ParticleType();
private:
	const int amountOfParticles;
	float* dev_x;
	float* dev_y;
	float* dev_z;
	float* dev_vx;
	float* dev_vy;
	float* dev_vz;
	float invmass;

	unsigned int vao, vbo;

	void setupDeviceData();
	void setupShaderData();
};

