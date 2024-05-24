#pragma once

#include "Renderer.hpp"

class ClothRenderer : public Renderer
{
public:
	explicit ClothRenderer(std::shared_ptr<Shader>& _shader);
	virtual void draw(const RenderInfo& drawData);
};