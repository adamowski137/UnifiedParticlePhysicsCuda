#include "ParticleRenderer.hpp"
#include "../Error/ErrorHandling.hpp"

ParticleRenderer::ParticleRenderer(std::shared_ptr<Shader>& _shader) : Renderer(_shader)
{
}

void ParticleRenderer::drawInstanced(const RenderInfo& drawData, int numInstances)
{
	glm::mat4 model = glm::scale(glm::vec3(sphereScale, sphereScale, sphereScale));
	shader->setUniformMat4fv("model", model);

	shader->bind();
	Call(glBindVertexArray(drawData.VAO));
	Call(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, drawData.IBO));
	Call(glDrawElementsInstanced(GL_TRIANGLES, drawData.numIndicies, GL_UNSIGNED_INT, nullptr, numInstances));
}

void ParticleRenderer::setSphereScale(float scale)
{
	this->sphereScale = scale;
}
