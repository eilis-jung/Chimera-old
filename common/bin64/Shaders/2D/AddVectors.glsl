#version 330 core

layout(location = 0) in vec2 velocityVec;
layout(location = 1) in vec2 centerVec;

uniform float scaleFactor;

out vec2 vOut;
void main()
{	
	vOut = scaleFactor*velocityVec + centerVec;
}