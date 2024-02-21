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
	Shader& getShader() { return *shader; }
};