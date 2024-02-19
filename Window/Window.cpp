#include "Window.hpp"
#include "glad/glad.h"
#include <functional>

Window::Window(int width, int height)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, "Unified Particle Physics Cuda", NULL, NULL);
    if (window == NULL)  throw std::bad_function_call();

    glfwMakeContextCurrent(window);
}


Window Window::getInstance(int width, int height)
{
	return Window(width, height);
}

void Window::runWindow()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}
