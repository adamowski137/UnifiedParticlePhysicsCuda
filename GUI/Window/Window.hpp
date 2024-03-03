#pragma once

#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include <memory>

#include "ImGuiOptions.hpp"

class Scene;

class GLFWwindowDeleter
{
public:
	void operator()(GLFWwindow* ptr)
	{
		glfwDestroyWindow(ptr);
	}
};

class Window
{
public:
	static Window Instance;
	void initInstance(int width, int height);
	Window(const Window& w) = delete;
	Window& operator=(const Window& other) = delete;
	const int width = 1024;
	const int height = 768;
	void clear(float r, float g, float b, float a);
	bool isClosed();
	void finishRendering(ImGuiOptions& options);
private:
	Window() {};
	void enableImGui();
	void renderImGui(ImGuiOptions& options);
	std::unique_ptr<GLFWwindow, GLFWwindowDeleter> glfw_window;
};