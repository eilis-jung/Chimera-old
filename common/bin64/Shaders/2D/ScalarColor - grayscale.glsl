#version 330 

#define ATTR_POSITION	0
#define ATTR_TEXCOORD	4
#define ATTR_INDEX		5

uniform float minScalar;
uniform float maxScalar;
uniform float avgScalar;


layout(location = ATTR_POSITION) in float scalarValue;

out float rColor;
out float gColor;
out float bColor;

void main()
{
	float totalDist = maxScalar - minScalar;

	rColor = (scalarValue - minScalar)/totalDist;
	gColor = (scalarValue - minScalar)/totalDist;
	bColor = (scalarValue - minScalar)/totalDist;
}
