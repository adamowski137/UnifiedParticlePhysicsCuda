#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <GLFW/glfw3.h>

class Camera
{
	glm::vec3 position;

	glm::vec3 cameraFront;
	glm::vec3 cameraUp;
	glm::vec3 cameraRight;
	glm::vec3 worldUp;
	const glm::mat4x4 projection;
public:
	Camera(int width, int height, glm::vec3 position);
	void updateVectors(glm::vec3 cameraDirection);
	void setPosition(glm::vec3 new_position);
	inline glm::vec3 getPosition() { return position; }
	glm::mat4x4 getProjectionViewMatrix();
	glm::vec3 getCameraFrontVector() { return cameraFront; };
};