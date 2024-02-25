class Constrain
{
public:
	Constrain(int n, float k, bool equality);
	~Constrain();
	float virtual operator()(int* index, float* x, float* y, float* z);

private:
	int n;
	float k;
	bool equality;
};