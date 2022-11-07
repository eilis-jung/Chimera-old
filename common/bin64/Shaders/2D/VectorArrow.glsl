#version 330 

// Declare all the semantics
#define ATTR_POSITION	0
#define ATTR_TEXCOORD	4
#define ATTR_INDEX		5

uniform float triangleLength;

layout(location = ATTR_POSITION)	in vec2 velocity;
layout(location = ATTR_TEXCOORD)	in vec2 gridCenter;

out vec2 v1Out;
out vec2 v2Out;
out vec2 v3Out;

void main()
{
	
	vec2 normalizedVel = normalize(velocity - gridCenter);
	float vecSize = length(velocity - gridCenter);
	float finalTriangleLength = 0;
	finalTriangleLength = clamp(triangleLength*log(1 + vecSize*10), 0.001, 0.008);
	

	//Calculate v1
	v1Out.x = -normalizedVel.y;
	v1Out.y = normalizedVel.x;
	v1Out *= finalTriangleLength/2;
	
	//Calculate v2
	v2Out.x = normalizedVel.y;
	v2Out.y = -normalizedVel.x;
	v2Out *= finalTriangleLength/2;
	
	//Calculate v3
	v3Out = finalTriangleLength*normalizedVel*0.866; 

	v1Out += velocity;
	v2Out += velocity;
	v3Out += velocity;
}
