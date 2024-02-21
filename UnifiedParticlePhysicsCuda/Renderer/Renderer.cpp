#include "Renderer.hpp"
#include "../Error/ErrorHandling.hpp"

Renderer::Renderer(std::shared_ptr<Shader>& _shader)
{
	shader = std::shared_ptr<Shader>(_shader);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
}

void Renderer::draw(const RenderInfo& drawData)
{
	shader->bind();
	Call(glBindVertexArray(drawData.VAO));
	Call(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, drawData.IBO));
	Call(glDrawElements(GL_TRIANGLES, drawData.numIndicies, GL_UNSIGNED_INT, nullptr));
}