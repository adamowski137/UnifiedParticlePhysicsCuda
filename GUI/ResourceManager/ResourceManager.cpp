#include "ResourceManager.hpp"
#include <filesystem>
#include <fstream>

#include "../Renderer/MeshGenerator.hpp"
#include "../Scene/TestScene/TestScene.hpp"
#include "../Scene/Scene_NaiveDraw/Scene_NaiveDraw.hpp"
#include "../Scene/Scene_Instancing/Scene_Instancing.hpp"

ResourceManager::ResourceManager()
{
	const std::string configPath = "../../../../GUI/res/config.txt";
	loadConfig(configPath);
}

ResourceManager& ResourceManager::get()
{
	static ResourceManager manager;
	return manager;
}

void ResourceManager::loadAllShaders(std::string resPath)
{
	for (const auto& entry : std::filesystem::directory_iterator(resPath))
	{
		if (std::filesystem::is_directory(entry))
		{
			std::string shaderName = entry.path().stem().string();
			Shaders.load(shaderName, (entry.path() / entry.path().stem()).string());
		}
	}
}

void ResourceManager::loadSphereData(int sectorCount, int stackCount)
{
	drawData.insert(std::make_pair("sphere", std::make_unique<RenderInfo>(getSphereData(sectorCount, stackCount))));
}

void ResourceManager::loadConfig(std::string configPath)
{
	std::ifstream file(configPath);

	file >> this->config.width;
	file >> this->config.height;

	file.close();
}

void ResourceManager::loadScenes()
{
	scenes.insert(std::make_pair("first scene", std::shared_ptr<Scene>(new TestScene())));
	scenes.insert(std::make_pair("naive drawing of lots of spheres", std::shared_ptr<Scene>(new Scene_NaiveDraw())));
	scenes.insert(std::make_pair("instanced drawing of lots of spheres", std::shared_ptr<Scene>(new Scene_Instancing())));
}
