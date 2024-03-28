#pragma once
#include <cuda_runtime.h>

struct Node
{
	int value;
	Node* next;
};

class List 
{
public:
	__host__ __device__ List();
	__host__ __device__ ~List();

	Node* head;
	Node* tail;

	__host__ __device__ void addNode(int value);
	__host__ __device__ void clearList();

private:
};
