class Constrain
{
public:
	Constrain(int n, float k, float cMin, float cMax);
	~Constrain();
private:
	int n;
	float k;
	float cMin;
	float cMax;
};