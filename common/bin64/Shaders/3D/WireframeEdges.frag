#version 130

in vec3 vBC;
float edgeFactor(){
	vec3 d = fwidth(vBC);
	vec3 a3 = smoothstep(vec3(0.0), d*0.95, vBC);
	return min(min(a3.x, a3.y), a3.z);
}

void main(){
	if(any(lessThan(vBC, vec3(0.02)))){
		gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
	}
	else{
		gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
	}
	gl_FragColor.rgb = mix(vec3(0.0), vec3(0.5), edgeFactor());

	gl_FragColor = vec4(0.0, 0.0, 0.0, (1.0-edgeFactor())*0.95);
	//gl_FragColor.a = 1.0;
}