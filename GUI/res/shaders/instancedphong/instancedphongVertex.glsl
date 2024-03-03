#version 450 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 offset;


uniform mat4 VP;
uniform mat4 model;
out vec3 normal;
out vec3 FragPos;

void main()
{
	normal = position;
	FragPos = vec3(model * vec4(position + offset, 1.0f));
	gl_Position = VP * vec4(FragPos, 1.0f);
}