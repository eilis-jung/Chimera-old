#version 330

in vec3 theColor;

out vec4 outputF;
void main()
{	
	outputF = vec4(theColor, 1.0f);
}