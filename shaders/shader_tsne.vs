#version 330 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aDer;
layout (location = 2) in int aLab;
layout (location = 3) in int aId;

out vec3 FragPos;  
out vec3 FragCol;  

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;


vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}


vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}



void main()
{
   gl_Position = projection * view * model * vec4(vec3(aPos.xy, 0.0f), 1.0f);
   FragPos = vec3(model * vec4(vec3(aPos.xy, 0.0f), 1.0f));

   float hue = float(aLab) / 10.0f;
   vec3 hsvCol = vec3(hue, 1.0f, 1.0f);

   //FragCol = vec3(1.0f);
   FragCol = hsv2rgb(hsvCol);
}
