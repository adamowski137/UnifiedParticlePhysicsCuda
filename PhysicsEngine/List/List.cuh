#include <cuda_runtime.h>

struct Node
{
	int value;
	Node* next;
};

class List 
{
public:
	List();
	~List();

	Node* head;
	Node* tail;

	__host__ __device__ void addNode(int value);
	__host__ __device__ void clearList();

private:
	int count;
};
