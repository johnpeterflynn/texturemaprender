#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in float Mask;

void main()
{    
	FragColor = vec4(TexCoords, Mask, 1.0);
}
