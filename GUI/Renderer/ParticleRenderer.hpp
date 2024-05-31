#pragma once
#include "Renderer.hpp"


class ParticleRenderer : public Renderer
{
	float sphereScale;
public:
	ParticleRenderer(std::shared_ptr<Shader>& _shader);
	void drawInstanced(const RenderInfo& drawData, int numInstances);
	void setSphereScale(float scale);
};