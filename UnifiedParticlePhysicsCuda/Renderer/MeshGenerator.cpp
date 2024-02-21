#include "MeshGenerator.hpp"

#include <vector>
#include <glad/glad.h>
#include <cmath>

RenderInfo getSphereData(int stackCount, int sectorCount)
{
	RenderInfo sphere;

	float sectorStep = 2 * 3.1415f / float(sectorCount), stackStep = 3.1415f / float(stackCount);
	float sectorAngle, stackAngle;

	std::vector<float> vertexData;
	vertexData.push_back(0.0f);
	vertexData.push_back(1.0f);
	vertexData.push_back(0.0f);

	for (int i = 1; i < stackCount; i++)
	{
		stackAngle = 3.1415f / 2.0f - i * stackStep;
		for (int j = 0; j < sectorCount; j++)
		{
			sectorAngle = j * sectorStep;
			vertexData.push_back(std::cosf(stackAngle) * std::cosf(sectorAngle));
			vertexData.push_back(std::sinf(stackAngle));
			vertexData.push_back(std::cosf(stackAngle) * std::sinf(sectorAngle));
		}
	}

	vertexData.push_back(0.0f);
	vertexData.push_back(-1.0f);
	vertexData.push_back(0.0f);
	std::vector<unsigned int> indicies;

	for (int j = 0; j < sectorCount; j++)
	{
		indicies.push_back(0);
		indicies.push_back(1 + j);
		indicies.push_back(1 + (j + 1) % sectorCount);
	}



	int k1, k2;
	for (int i = 0; i < stackCount - 2; i++)
	{
		k1 = i * sectorCount + 1;
		k2 = k1 + sectorCount;
		for (int j = 0; j < sectorCount; j++)
		{
			indicies.push_back(k1 + j);
			indicies.push_back(k2 + j);
			indicies.push_back(k2 + (j + 1) % sectorCount);
			indicies.push_back(k2 + (j + 1) % sectorCount);
			indicies.push_back(k1 + (j + 1) % sectorCount);
			indicies.push_back(k1 + j);
		}
	}

	for (int j = 0; j < sectorCount; j++)
	{

		indicies.push_back(vertexData.size() / 3 - 1);
		indicies.push_back(k2 + j);
		indicies.push_back(k2 + (j + 1) % sectorCount);
	}

	sphere.generate(vertexData, indicies, { {3, GL_FLOAT} });
	return sphere;
}