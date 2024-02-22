#include "Camera.hpp"

void Camera::updateVectors(glm::vec3 direction)
{
    cameraFront = glm::normalize(direction);
    cameraRight = glm::normalize(glm::cross(cameraFront, worldUp));
    cameraUp = glm::normalize(glm::cross(cameraRight, cameraFront));
}


Camera::Camera(int width, int height) : 
    projection(glm::perspective(glm::radians(45.0f), static_cast<float>(width) / static_cast<float>(height), 0.01f, 200.0f))
{
    cameraFront = glm::vec3(0.0f, 0.0f, 1.0f);
    cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    worldUp = cameraUp;
    cameraRight = glm::normalize(glm::cross(cameraFront, worldUp));
}


glm::mat4x4 Camera::getProjectionViewMatrix(glm::vec3 position)
{
    return projection * glm::lookAt(position, position + cameraFront, cameraUp);
}