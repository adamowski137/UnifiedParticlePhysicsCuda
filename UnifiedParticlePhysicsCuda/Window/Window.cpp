#include "Window.hpp"
#include <functional>

Window::Window()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfw_window = std::unique_ptr<GLFWwindow, GLFWwindowDeleter>(glfwCreateWindow(width, height, "Unified Particle Physics Cuda", NULL, NULL));
    if (glfw_window == NULL)  throw std::bad_function_call();

    glfwMakeContextCurrent(glfw_window.get());
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        throw "Failed to initialize GLAD";
    }
    gladLoadGL();
}

void Window::clear(float r, float g, float b, float a)
{
    glClearColor(r / 255.0f, g / 255.0f, b / 255.0f, a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

bool Window::isClosed()
{
    return glfwWindowShouldClose(glfw_window.get());
}

void Window::finishRendering()
{
    glfwSwapBuffers(glfw_window.get());
    glfwPollEvents();
}


Window& Window::getInstance()
{
    static Window window;
    return window;
}