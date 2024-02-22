#version 450 core

out vec4 out_color;
in vec3 normal;
in vec3 FragPos;

uniform vec3 lightPos;
uniform vec3 cameraPos;
uniform vec3 color;

void main()
{
	// ambient
	vec3 ambient = vec3(0.1f, 0.1f, 0.1f);

	// diffuse
	vec3 lightDir = normalize(lightPos - FragPos);
	float diff = max(dot(normal, lightDir), 0.0f);

	// specular
	float specularStrength = 0.5f;
	vec3 viewDir = normalize(cameraPos - FragPos);
	vec3 reflectDir = reflect(-lightDir, normal);
	float spec = pow(max(dot(viewDir, reflectDir), 0.0f), 32);

	out_color = vec4(color * (ambient + vec3(1.0f, 1.0f, 1.0f) * (diff + spec * specularStrength)), 1.0f);
}