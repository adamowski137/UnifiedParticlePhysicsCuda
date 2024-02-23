#pragma once
#include <vector>
#include <string>


class RenderInfo
{
public:
	unsigned int VAO;
	unsigned int IBO;
	unsigned int numIndicies;
	RenderInfo() : VAO(0), IBO(0), numIndicies(0) {};
	RenderInfo(unsigned int _VAO, unsigned int _IBO, unsigned int _numIndicies) : VAO(_VAO), IBO(_IBO), numIndicies(_numIndicies) {};

	void generate(std::vector<float> verticiesData, std::vector<unsigned int> indiciesData, std::vector<std::pair<unsigned, unsigned>> structure);
	void addInstancing(std::vector<float> offsetData);
};