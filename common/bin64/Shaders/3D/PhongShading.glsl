
#version 130
uniform vec3 lightPos;

varying vec3 N;
varying vec3 v;

out vec3 vBC;
void main(void)
{
	float modB = mod(gl_VertexID, 3);

	if (modB == 0.0f)
		vBC = vec3(1.0f, 0.0f, 0.0f);
	else if (modB == 1.0f)
		vBC = vec3(0.0f, 1.0f, 0.0f);
	else
		vBC = vec3(0.0f, 0.0f, 1.0f);

	v = vec3(gl_ModelViewMatrix * gl_Vertex);
	N = normalize(gl_NormalMatrix * gl_Normal);
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
