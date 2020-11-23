#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec4 aVertexColor;
//layout (location = 4) in float aMask;

out vec4 FragmentColor;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragmentColor = aVertexColor;
    Normal = aNormal;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
