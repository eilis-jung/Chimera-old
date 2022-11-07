#version 330 

#define ATTR_POSITION	0
#define ATTR_TEXCOORD	4
#define ATTR_INDEX		5

uniform float minPressure;
uniform float maxPressure;
uniform float avgPressure;


layout(location = ATTR_POSITION) in float pressure;

out float rColor;
out float gColor;
out float bColor;

void main()
{
	float gamma = 1.0f;
	float factor = 0.0f;

	float totalDist = maxPressure - minPressure;
	float wavelength = 420 + ((pressure - minPressure)/totalDist)*360;
	
	if(wavelength <= 439){
		rColor	  = -(wavelength - 440) / (440.0f - 350.0f);
		gColor = 0.0;
		bColor  = 1.0;
	} else if(wavelength <= 489){
		rColor	= 0.0;
		gColor = (wavelength - 440) / (490.0f - 440.0f);
		bColor	= 1.0;
	} else if(wavelength <= 509){
		rColor = 0.0;
		gColor = 1.0;
		bColor = -(wavelength - 510) / (510.0f - 490.0f);
	} else if(wavelength <= 579){ 
		rColor = (wavelength - 510) / (580.0f - 510.0f);
		gColor = 1.0;
		bColor = 0.0;
	} else if(wavelength <= 644){
		rColor = 1.0;
		gColor = -(wavelength - 645) / (645.0f - 580.0f);
		bColor = 0.0;
	} else if(wavelength <= 780){
		rColor = 1.0;
		gColor = 0.0;
		bColor = 0.0;
	} else {
		rColor = 0.0;
		gColor = 0.0;
		bColor = 0.0;
	}


}
