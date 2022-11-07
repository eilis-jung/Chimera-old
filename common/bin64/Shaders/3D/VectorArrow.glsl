#version 330 

// Declare all the semantics
#define ATTR_POSITION	0
#define ATTR_TEXCOORD	4
#define ATTR_INDEX		5

uniform float triangleLength;
uniform vec3 rotationVec;

layout(location = ATTR_POSITION)	in vec3 velocity;
layout(location = ATTR_TEXCOORD)	in vec3 gridCenter;

out vec3 v1Out;
out vec3 v2Out;
out vec3 v3Out;

void main()
{
	
	vec3 normalizedVel = normalize(velocity - gridCenter);
	float vecSize = length(velocity - gridCenter);
	float finalTriangleLength = 0;
	finalTriangleLength = clamp(triangleLength*log(1 + vecSize*10), 0.001, 0.008);
	

	//Calculate v1
	v1Out = cross(normalizedVel, rotationVec);
	v1Out *= finalTriangleLength/2;
	
	//Calculate v2
	v2Out = cross(normalizedVel, -rotationVec);
	v2Out *= finalTriangleLength/2;
	
	//Calculate v3
	v3Out = finalTriangleLength*normalizedVel*0.866; 

	v1Out += velocity;
	v2Out += velocity;
	v3Out += velocity;
}
