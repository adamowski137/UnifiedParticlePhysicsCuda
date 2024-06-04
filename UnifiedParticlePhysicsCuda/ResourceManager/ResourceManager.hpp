#pragma once
#include "SharedResourceHolder.hpp"
#include "../../GUI/Shader/Shader.hpp"
#include "../../GUI/Window/ImGuiOptions.hpp"
#include "../Scene/Scene.hpp"
#include "../../PhysicsEngine/Config/Config.hpp"

#include "Config.h"

#include <string>
#include <vector>

class ResourceManager
{
private:
	ResourceManager();
	std::shared_ptr<Scene> currentScene;
	std::unordered_map<std::string, EngineConfig> configs;
public:
	static ResourceManager Instance;
	void loadAllShaders(std::string resPath);
	void loadSphereData(int sectorCount, int stackCount);
	void loadConfig(std::string configPath);
	void loadScenes(int amountOfPoints);

	std::shared_ptr<Scene>& getActiveScene();

	ResourceManager(const ResourceManager& other) = delete;
	ResourceManager& operator=(const ResourceManager& other) = delete;
	ImGuiOptions options;
	Config config;
	std::unordered_map<std::string, std::unique_ptr<RenderInfo>> drawData;
	std::unordered_map<std::string, std::shared_ptr<Scene>> scenes;
	SharedResourceManager<Shader> Shaders;
};