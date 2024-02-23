#include "Window.hpp"
#include <functional>
#include "../imgui/imgui.h"
#include "../imgui/backends/imgui_impl_glfw.h"
#include "../imgui/backends/imgui_impl_opengl3.h"
#include "../Error/ErrorHandling.hpp"
#include "../ResourceManager/ResourceManager.hpp"

Window::Window()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfw_window = std::unique_ptr<GLFWwindow, GLFWwindowDeleter>(glfwCreateWindow(
        ResourceManager::get().config.width,
        ResourceManager::get().config.height,
        "Unified Particle Physics Cuda", NULL, NULL));

    if (glfw_window == NULL)  throw std::bad_function_call();

    glfwMakeContextCurrent(glfw_window.get());
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        throw "Failed to initialize GLAD";
    }
    gladLoadGL();
    enableImGui();

    Call(glEnable(GL_DEPTH_TEST));
    Call(glEnable(GL_CULL_FACE));
    // TODO: naprawic 
    Call(glFrontFace(GL_CW));
    Call(glCullFace(GL_BACK));
}

void Window::enableImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io{ ImGui::GetIO() }; (void)io;
    
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    
    ImGui_ImplGlfw_InitForOpenGL(glfw_window.get(), true);
    ImGui_ImplOpenGL3_Init("#version 450 core");
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

void Window::renderImGui(std::shared_ptr<Scene>& currScene)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiIO& io{ ImGui::GetIO() }; (void)io;

    ImGui::Begin("Choose scene");
    for (const auto& it : ResourceManager::get().scenes)
    {
        if (ImGui::RadioButton(it.first.c_str(), currScene.get() == it.second.get()))
        {
            currScene = std::shared_ptr<Scene>(it.second);
        }
    }

    ImGui::Separator();
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Window::finishRendering(std::shared_ptr<Scene>& currScene)
{
    renderImGui(currScene);
    glfwSwapBuffers(glfw_window.get());
    glfwPollEvents();
}


Window& Window::getInstance()
{
    static Window window;
    return window;
}