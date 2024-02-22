#include "RenderInfo.hpp"

#include <numeric>
#include <functional>

#include <glad/glad.h>

void RenderInfo::generate(std::vector<float> verticiesData, std::vector<unsigned int> indiciesData, std::vector<std::pair<unsigned, unsigned>> structure)
{
    glGenVertexArrays(1, &this->VAO);
    unsigned VBO;
    glGenBuffers(1, &VBO);
    glBindVertexArray(this->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * verticiesData.size(), verticiesData.data(), GL_STATIC_DRAW);
    unsigned vertexDataSize = std::accumulate(std::begin(structure), std::end(structure), 0, [](unsigned x, std::pair<unsigned, unsigned> p) {
        return x + p.first;
        });
    unsigned vertexDataOffset = 0;
    for (int i = 0; i < structure.size(); i++)
    {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, structure[i].first, structure[i].second, GL_FALSE, vertexDataSize * sizeof(float), (const void*)(sizeof(float) * vertexDataOffset));
        vertexDataOffset += structure[i].first;
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glGenBuffers(1, &this->IBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned) * indiciesData.size(), indiciesData.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
    this->numIndicies = indiciesData.size();
}