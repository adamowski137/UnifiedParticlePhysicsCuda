#pragma once
#include <vector>
#include <string>

struct SceneData
{
	std::string name;
	bool isActive = false;
};

struct ImGuiOptions
{
	std::vector<SceneData> sceneData;
	bool sceneChanged = false;
};