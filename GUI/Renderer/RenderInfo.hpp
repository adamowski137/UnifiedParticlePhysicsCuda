#pragma once
#include <vector>
#include <string>

class RenderInfo
{
public:
	unsigned int VAO;
	unsigned int VBO;
	unsigned int IBO;
	unsigned int numIndicies;
	unsigned int instancingVBO;
	RenderInfo() : VAO(0), IBO(0), VBO(0), instancingVBO(0), numIndicies(0) {};
	

	void generate(std::vector<float> verticiesData, std::vector<unsigned int> indiciesData, std::vector<std::pair<unsigned, unsigned>> structure);
	void addInstancing(std::vector<float> offsetData);
	void dispose();
};

class RenderInfoDeleter
{
public:
	void operator()(RenderInfo* obj)
	{
		obj->dispose();
	}
};