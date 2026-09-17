#version 330 core

out vec4 FragColor;
  
in vec3 FragPos;

float rand(vec2 co)
{
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main()
{
	float valx = rand(FragPos.xy);
	float valy = rand(FragPos.xz);
	float valz = rand(FragPos.zy);

	//float valx = (atan(length(FragPos.z)/1.0f) / 3.14f) + 0.5f;
	//float valy = (atan(length(FragPos.z)/10.0f) / 3.14f) + 0.5f;
	//float valz = (atan(length(FragPos.z)/5.0f) / 3.14f) + 0.5f;
	FragColor = vec4(vec3((valx),(valy),(valz)),1.0f);
	//FragColor = vec4(1.0f);
}