#version 330 core

out vec4 FragColor;
  
in vec3 FragPos;
in vec3 FragVel;
in vec3 FragCol;
flat in int FragMass;

void main()
{
	vec2 coord = gl_PointCoord - vec2(0.5);
	//if(length(coord) > 0.2)
	if(length(coord) > 0.15)
	{
		discard;
	}

	FragColor = vec4(FragCol,1.0f);
}
