#pragma once

#include "GLFW/glfw3.h"

class Window
{
public:
	static Window getInstance(int width, int height);

	void runWindow();

private:
	GLFWwindow* window;
	
	Window(int width, int height);
	static int width, height;

};