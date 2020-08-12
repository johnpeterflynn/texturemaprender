#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec4 gl_FragCoord;

uniform sampler2D texture_diffuse1;

void main()
{    
//  FragColor = texture(texture_diffuse1, TexCoords);
	FragColor = vec4(TexCoords, (gl_FragCoord.z > 0.0), 1.0);

}
