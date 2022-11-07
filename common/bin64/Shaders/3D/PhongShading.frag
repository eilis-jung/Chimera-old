#version 130
uniform vec3 lightPos;

varying vec3 N;
varying vec3 v;

const vec3 diffuseColor = vec3(1.00, 0.50, 0.50);
const vec3 specColor = vec3(1.0, 1.0, 1.0);
const vec3 lightPos = vec3(3.0, 3.0, 6.0);

in vec3 vBC;
float edgeFactor(){
	vec3 d = fwidth(vBC);
	vec3 a3 = smoothstep(vec3(0.0), d*0.95, vBC);
	return min(min(a3.x, a3.y), a3.z);
}

void main() {

  vec3 normal = normalize(N); 
  vec3 lightDir = normalize(lightPos - v);

  float lambertian = max(dot(lightDir,normal), 0.0);
  float specular = 0.0;

  if(lambertian > 0.0) {

    vec3 reflectDir = reflect(-lightDir, normal);
    vec3 viewDir = normalize(-v);

    float specAngle = max(dot(reflectDir, viewDir), 0.0);
    specular = pow(specAngle, 4.0);
	specular = 0;
  }
  specular = 0;
  gl_FragColor.rgb = lambertian*diffuseColor + specular*specColor;
}
