#version 130

uniform int trailSize;
uniform int maxNumberOfParticles;

out vec3 theColor;

void main()
{	
	int lowerIndexBound = gl_VertexID / maxNumberOfParticles;
	float floatingIndex = lowerIndexBound / float(trailSize);

	vec3 initialColor = vec3(0.0);
	vec3 finalColor = vec3(1.0);

	theColor = mix(initialColor, finalColor, floatingIndex);

	gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
}