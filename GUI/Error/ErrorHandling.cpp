#include "ErrorHandling.hpp"
#include <glad/glad.h>
#include <iostream>

void glClearError()
{
    while (glGetError() != GL_NO_ERROR);
}

bool glCheckError()
{
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR)
    {
        std::cout << "Error: " << error << std::endl;
        __debugbreak();
    }
    return true;
}