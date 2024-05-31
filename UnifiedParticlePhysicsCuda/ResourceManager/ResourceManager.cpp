#include "ResourceManager.hpp"
#include <filesystem>
#include <fstream>

#include "../../GUI/Renderer/MeshGenerator.hpp"
#include "../Scene/TestScene/TestScene.cuh"
#include "../Scene/Scene_Covering/Scene_Covering.cuh"
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
	scenes.insert(std::make_pair("external scene", std::shared_ptr<Scene>(new Scene_External(amountOfPoints))));
	scenes.insert(std::make_pair("first scene", std::shared_ptr<Scene>(new TestScene(5000))));
	scenes.insert(std::make_pair("Cloth simulation", std::shared_ptr<Scene>(new Cloth_Scene())));
	scenes.insert(std::make_pair("Rigid body simulation", std::shared_ptr<Scene>(new Scene_RigidBody())));
	scenes.insert(std::make_pair("Covering simulaiton", std::shared_ptr<Scene>(new Scene_Covering())));
	//scenes.insert(std::make_pair("naive drawing of lots of spheres", std::shared_ptr<Scene>(new Scene_NaiveDraw())));

	for (const auto& it : scenes)
	{
		SceneData data;
		data.name = it.first;
		data.isActive = false;
		options.sceneData.push_back(data);
	}

	options.sceneData[0].isActive = true;
	currentScene = scenes[options.sceneData[0].name];
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
				scenes[options.sceneData[i].name].get()->reset();
				break;
			}
		}
		options.sceneChanged = false;
	}

	return currentScene;
}
