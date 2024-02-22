#pragma once
#include "../Shader/Shader.hpp"
#include "RenderInfo.hpp"

#include <memory>

class Renderer
{
	std::shared_ptr<Shader> shader;
public:
	explicit Renderer(std::shared_ptr<Shader>& _shader);
	void draw(const RenderInfo& drawData);
	void draw(const RenderInfo& drawData, glm::vec3 position, glm::vec3 color);
	void setCameraPosition(glm::vec3 cameraPos);
	void setLightSourcePosition(glm::vec3 lightSourcePos);
	Shader& getShader() { return *shader; }
};