#include "TestScene.hpp"
#include "../../ResourceManager/ResourceManager.hpp"

TestScene::TestScene() : Scene(ResourceManager::Instance.Shaders["phong"])
{

}

TestScene::~TestScene()
{
}

void TestScene::update()
{
	glm::vec3 cameraPos = { 0.f, 0.f, -10.f };
	renderer->getShader().setUniformMat4fv("VP", camera.getProjectionViewMatrix(cameraPos));
	renderer->setCameraPosition(cameraPos);
	renderer->setLightSourcePosition({ 5 * std::cos(glfwGetTime()), 0, 5 * std::sin(glfwGetTime()) });
	camera.updateVectors({ 0.f, 0.f, 1.f });
}

void TestScene::draw()
{
	renderer->draw(*ResourceManager::Instance.drawData["sphere"], { 0, 0, 0 }, { 1, 1, 0 });
}
