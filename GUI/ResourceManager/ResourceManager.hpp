#pragma once
#include "SharedResourceHolder.hpp"
#include "../Shader/Shader.hpp"
#include "../Renderer/RenderInfo.hpp"
#include "../Scene/Scene.hpp"

#include "Config.h"

#include <string>

class ResourceManager
{
private:
	ResourceManager();
public:
	static ResourceManager& get();
	void loadAllShaders(std::string resPath);
	void loadSphereData(int sectorCount, int stackCount);
	void loadConfig(std::string configPath);
	void loadScenes(int amountOfPoints);

	ResourceManager(const ResourceManager& other) = delete;
	ResourceManager& operator=(const ResourceManager& other) = delete;

	std::unordered_map<std::string, std::unique_ptr<RenderInfo>> drawData;
	std::unordered_map<std::string, std::shared_ptr<Scene>> scenes;
	SharedResourceManager<Shader> Shaders;
	Config config;
};