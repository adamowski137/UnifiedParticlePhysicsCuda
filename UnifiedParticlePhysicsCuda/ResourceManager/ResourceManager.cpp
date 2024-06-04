#include "ResourceManager.hpp"
#include <filesystem>
#include <fstream>

#include "../../GUI/Renderer/MeshGenerator.hpp"
#include "../Scene/TestScene/TestScene.cuh"
#include "../Scene/Scene_Covering/Scene_Covering.cuh"
#include "../Scene/Scene_Trampoline/Scene_Trampoline.cuh"
#include "../Scene/Scene_External/Scene_External.cuh"
#include "../Scene/Cloth_Scene/Cloth_Scene.cuh"
#include "../Scene/RigidBody_Scene/Scene_RigidBody.cuh"

ResourceManager ResourceManager::Instance;

ResourceManager::ResourceManager()
{
	const std::string configPath = "../../../../GUI/res/config.txt";
	loadConfig(configPath);
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

void ResourceManager::loadScenes(int amountOfPoints)
{
	const std::string scenesPath = "../../../../UnifiedParticlePhysicsCuda/Scene/";
	
	EngineConfig currConfig = EngineConfig::readConfig(scenesPath + "Scene_External/config.txt");
	GlobalEngineConfig::config = currConfig;
	configs.insert(std::make_pair("external scene", currConfig));	
	scenes.insert(std::make_pair("external scene", std::shared_ptr<Scene>(new Scene_External(amountOfPoints))));

	currConfig = EngineConfig::readConfig(scenesPath + "TestScene/config.txt");
	GlobalEngineConfig::config = currConfig;
	scenes.insert(std::make_pair("first scene", std::shared_ptr<Scene>(new TestScene(5000))));
	configs.insert(std::make_pair("first scene", currConfig));

	currConfig = EngineConfig::readConfig(scenesPath + "Cloth_Scene/config.txt");
	GlobalEngineConfig::config = currConfig;
	scenes.insert(std::make_pair("Cloth simulation", std::shared_ptr<Scene>(new Cloth_Scene())));
	configs.insert(std::make_pair("Cloth simulation", currConfig));
	
	currConfig = EngineConfig::readConfig(scenesPath + "RigidBody_Scene/config.txt");
	GlobalEngineConfig::config = currConfig;
	scenes.insert(std::make_pair("Rigid body simulation", std::shared_ptr<Scene>(new Scene_RigidBody())));
	configs.insert(std::make_pair("Rigid body simulation", currConfig));
	
	currConfig = EngineConfig::readConfig(scenesPath + "Scene_Covering/config.txt");
	GlobalEngineConfig::config = currConfig;
	scenes.insert(std::make_pair("Covering simulation", std::shared_ptr<Scene>(new Scene_Covering())));
	configs.insert(std::make_pair("Covering simulation", currConfig));

	currConfig = EngineConfig::readConfig(scenesPath + "Scene_Trampoline/config.txt");
	GlobalEngineConfig::config = currConfig;
	scenes.insert(std::make_pair("Cloth trampoline simulation", std::shared_ptr<Scene>(new Scene_Trampoline())));
	configs.insert(std::make_pair("Cloth trampoline simulation", currConfig));

	for (const auto& it : scenes)
	{
		SceneData data;
		data.name = it.first;
		data.isActive = false;
		options.sceneData.push_back(data);
	}

	options.sceneData[3].isActive = true;
	currentScene = scenes[options.sceneData[3].name];
	GlobalEngineConfig::config = configs[options.sceneData[3].name];
	currentScene.get()->reset();
}

std::shared_ptr<Scene>& ResourceManager::getActiveScene()
{
	if (options.sceneChanged)
	{
		for (int i = 0; i < options.sceneData.size(); i++)
		{
			if (options.sceneData[i].isActive)
			{
				currentScene = scenes[options.sceneData[i].name];
				GlobalEngineConfig::config = configs[options.sceneData[i].name];
				currentScene.get()->reset();
				break;
			}
		}
		options.sceneChanged = false;
	}

	return currentScene;
}
