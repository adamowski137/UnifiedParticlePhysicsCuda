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
	static Window& getInstance(int width, int height);
	void runWindow();
	Window(const Window& w) = delete;
	Window& operator=(const Window& other) = delete;
private:
	Window(int width, int height);
	std::unique_ptr<GLFWwindow, GLFWwindowDeleter> glfw_window;
	void clear(float r, float g, float b, float a);
	static int width, height;

};