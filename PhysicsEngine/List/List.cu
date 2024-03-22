#include "List.cuh"
#include <stdlib.h>

List::List() : head{ NULL }, tail{ NULL }
{
}

List::~List()
{
	//clearList();
}

void List::addNode(int value)
{
	Node* p = (Node*) malloc(sizeof(Node));
	p->value = value;
	p->next = NULL;
	if (head == NULL)
	{
		head = p;
		tail = p;
		return;
	}
	
	tail->next = p;
	tail = tail->next;
}

void List::clearList()
{
	if (head == NULL) return;
	Node* p = head;
	while (p != NULL)
	{
		Node* pp = p;
		p = p->next;
		free(pp);
	}
	head = NULL;
	tail = NULL;
}