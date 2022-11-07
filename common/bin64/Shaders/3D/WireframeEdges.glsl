#version 130

/** Simple wireframe shader. Only works for triangle meshes */
out vec3 vBC;
in float barycentric;

void main(){
	float modB = mod(gl_VertexID, 3);
	if (modB == 0.0f)
		vBC = vec3(1.0f, 0.0f, 0.0f);
	else if (modB == 1.0f)
		vBC = vec3(0.0f, 1.0f, 0.0f);
	else
		vBC = vec3(0.0f, 0.0f, 1.0f);
	//vBC = barycentric;
	gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
}