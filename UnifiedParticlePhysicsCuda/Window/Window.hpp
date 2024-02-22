#pragma once

#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include <memory>


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
	static Window& getInstance();
	Window(const Window& w) = delete;
	Window& operator=(const Window& other) = delete;
	const int width = 1024;
	const int height = 768;
	void clear(float r, float g, float b, float a);
	bool isClosed();
	void finishRendering();
private:
	Window();
	void enableImGui();
	std::unique_ptr<GLFWwindow, GLFWwindowDeleter> glfw_window;
};