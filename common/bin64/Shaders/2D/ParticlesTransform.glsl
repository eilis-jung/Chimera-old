uniform float gridSpacing;
void main()
{
	vec4 position = gl_Vertex;
	position.x = gl_Vertex.x*gridSpacing;
	position.y = gl_Vertex.y*gridSpacing;	
	gl_Position = gl_ModelViewProjectionMatrix*position;
}
