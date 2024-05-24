#include "ClothRenderer.hpp"
#include "../Error/ErrorHandling.hpp"

ClothRenderer::ClothRenderer(std::shared_ptr<Shader>& _shader) : Renderer(_shader)
{
}

void ClothRenderer::draw(const RenderInfo& drawData)
{
	glm::mat4 model = glm::scale(glm::vec3(0.1f, 0.1f, 0.1f));
	shader->setUniformMat4fv("model", model);

	shader->bind();
	Call(glBindVertexArray(drawData.VAO));
	Call(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, drawData.IBO));
	Call(glDrawElements(GL_LINES, drawData.numIndicies, GL_UNSIGNED_INT, nullptr));
}
