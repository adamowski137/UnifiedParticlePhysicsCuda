#version 450 core

layout(location = 0) in vec3 position;


uniform mat4 VP;
uniform mat4 model;
out vec3 normal;
out vec3 FragPos;

void main()
{
	gl_Position = VP * model * vec4(position, 1.0f);
	normal = position;
	FragPos = vec3(model * vec4(position, 1.0f));
}