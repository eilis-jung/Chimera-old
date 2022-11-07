#version 330 core

layout(location = 0) in vec3 v1;
layout(location = 1) in vec3 v2;
uniform float scaleFactor;
out vec3 vOut;

void main()
{	
	vOut = scaleFactor*v1 + v2;
}