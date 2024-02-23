#include "Renderer.hpp"
#include "../Error/ErrorHandling.hpp"

Renderer::Renderer(std::shared_ptr<Shader>& _shader)
{
	shader = std::shared_ptr<Shader>(_shader);
}

void Renderer::draw(const RenderInfo& drawData)
{
	shader->bind();
	Call(glBindVertexArray(drawData.VAO));
	Call(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, drawData.IBO));
	Call(glDrawElements(GL_TRIANGLES, drawData.numIndicies, GL_UNSIGNED_INT, nullptr));
}

void Renderer::drawInstanced(const RenderInfo& drawData, int numInstances)
{
	glm::mat4 model = glm::mat4x4(1.f);
	shader->setUniformMat4fv("model", model);

	shader->setUniform3f("color", glm::vec3(1.f, 0.f, 1.f));

	shader->bind();
	Call(glBindVertexArray(drawData.VAO));
	Call(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, drawData.IBO));
	Call(glDrawElementsInstanced(GL_TRIANGLES, drawData.numIndicies, GL_UNSIGNED_INT, nullptr, numInstances));
}

void Renderer::draw(const RenderInfo& drawData, glm::vec3 position, glm::vec3 color)
{
	glm::mat4 model = glm::translate(position);
	shader->setUniformMat4fv("model", model);

	shader->setUniform3f("color", color);

	this->draw(drawData);
}

void Renderer::setCameraPosition(glm::vec3 cameraPos)
{
	shader->setUniform3f("cameraPos", cameraPos);
}

void Renderer::setLightSourcePosition(glm::vec3 lightSourcePos)
{
	shader->setUniform3f("lightPos", lightSourcePos);
}
