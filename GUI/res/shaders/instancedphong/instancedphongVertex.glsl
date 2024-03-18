#version 450 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 offset;


uniform mat4 VP;
uniform mat4 model;

out vec3 normal;
out vec3 FragPos;
out vec3 inst_color;

vec3 colors[3] = vec3[3](
	vec3(1.f, 0.f, 0.f),
	vec3(0.f, 1.f, 0.f),
	vec3(0.f, 0.f, 1.f)
);


void main()
{
	normal = position;
	inst_color = colors[gl_InstanceID % 3];
	FragPos = vec3(model * vec4(position + offset, 1.0f));
	gl_Position = VP * vec4(FragPos, 1.0f);
}