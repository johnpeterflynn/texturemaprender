#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in float Mask;

void main()
{    
	//FragColor = vec4(TexCoords, Mask, 1.0);
	FragColor = vec4(0.0, 0.0, Mask, 1.0);
	//FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
