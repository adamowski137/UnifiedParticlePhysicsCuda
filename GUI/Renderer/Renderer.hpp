#pragma once
#include "../Shader/Shader.hpp"
#include "RenderInfo.hpp"

#include <memory>

class Renderer
{
	std::shared_ptr<Shader> shader;
	float sphereScale;
	glm::vec3 sphereColor;
public:
	explicit Renderer(std::shared_ptr<Shader>& _shader);
	void draw(const RenderInfo& drawData);
	void drawInstanced(const RenderInfo& drawData, int numInstances);
	void draw(const RenderInfo& drawData, glm::vec3 position, glm::vec3 color);
	void setCameraPosition(glm::vec3 cameraPos);
	void setLightSourcePosition(glm::vec3 lightSourcePos);

	void setSphereScale(float scale);
	void setSphereColor(glm::vec3 color);
	Shader& getShader() { return *shader; }
};