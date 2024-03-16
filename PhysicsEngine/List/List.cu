#include "List.cuh"
#include <stdlib.h>

List::List() : count{ 0 }, head{ NULL }, tail{ NULL }
{
}

List::~List()
{
	clearList();
}

void List::addNode(int value)
{
	Node* p = (Node*) malloc(sizeof(Node));
	p->value = value;
	count++;
	if (head == NULL)
	{
		head = p;
		tail = p;
		return;
	}
	
	tail->next = p;
}

void List::clearList()
{
	Node* p = head;
	while (p != NULL)
	{
		Node* pp = p->next;
		free(p);
	}
	count = 0;
}