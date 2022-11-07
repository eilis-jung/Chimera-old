#include "Rendering/ParticleSystem3D.h"


namespace Chimera {

	void ParticleSystem3D::initializeParticles() {
		m_pParticlesPosition = new Vector3[m_params.numParticles];
		m_pParticlesVelocity = new Vector3[m_params.numParticles];
		m_pParticlesLifeTime = new Scalar[m_params.numParticles];
		m_pParticlesVorticity = new Scalar[m_params.numParticles];
		m_pParticlesColor = new Vector3[m_params.numParticles];
		m_EmitterInitialPosition = m_params.EmitterPosition;

		for(int i = 0; i < m_params.numParticles; i++) {
			m_pParticlesPosition[i].x = FLT_MIN;
			m_pParticlesPosition[i].y = FLT_MIN;
			m_pParticlesPosition[i].z = FLT_MIN;
			m_pParticlesVelocity[i] = Vector3(0, 0, 0);
			m_pParticlesLifeTime[i] = FLT_MAX;
			m_pParticlesVorticity[i] = 0;
			m_pParticlesColor[i] = Vector3(1, 0, 0); //All red, just for testing
		}

		if(m_params.EmitterType == squareRandomEmitter) {
			for(int i = 0; i < m_params.initialNumParticles; i++) {
				m_pParticlesPosition[i].x = m_params.EmitterPosition.x + m_params.EmitterSize.x*(rand()/(float) RAND_MAX);
				m_pParticlesPosition[i].y = m_params.EmitterPosition.y + m_params.EmitterSize.y*(rand()/(float) RAND_MAX);
				m_pParticlesPosition[i].z = m_params.EmitterPosition.z + m_params.EmitterSize.z*(rand()/(float) RAND_MAX);
				m_pParticlesVelocity[i] = Vector3(0, 0, 0);
				m_pParticlesLifeTime[i] = m_params.particlesLife + m_params.particlesLifeVariance*(rand()/(float) RAND_MAX);
				m_pParticlesVorticity[i] = 0;
			}
		} else if(m_params.EmitterType == circleRandomEmitter) {
			for(int i = 0; i < m_params.initialNumParticles; i++) {
				m_pParticlesPosition[i].x = m_params.EmitterPosition.x + m_params.EmitterSize.x*(rand()/(float) RAND_MAX);
				m_pParticlesPosition[i].y = m_params.EmitterPosition.y + m_params.EmitterSize.y*(rand()/(float) RAND_MAX);
				m_pParticlesPosition[i].z = m_params.EmitterPosition.z + m_params.EmitterSize.z*(rand()/(float) RAND_MAX);
				m_pParticlesVelocity[i] = Vector3(0, 0, 0);
				m_pParticlesLifeTime[i] = m_params.particlesLife + m_params.particlesLifeVariance*(rand()/(float) RAND_MAX);
				m_pParticlesVorticity[i] = 0;
			}
		}
		numInitializedParticles = m_params.initialNumParticles;
	}
	void ParticleSystem3D::initializeVBOs() {
		m_pParticlesVBO = new GLuint();
		unsigned int sizeParticlesVBO = m_params.numParticles*sizeof(Vector3);

		glGenBuffers(1, m_pParticlesVBO);
		glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeParticlesVBO, m_pParticlesPosition, GL_DYNAMIC_DRAW);

		m_pParticlesColorVBO = new GLuint;
		unsigned int sizeParticlesColorVBO = m_params.numParticles*sizeof(Vector3);
		glGenBuffers(1, m_pParticlesColorVBO);
		glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeParticlesColorVBO, m_pParticlesColor, GL_DYNAMIC_DRAW);
	}

	void ParticleSystem3D::updateLocalAxis() {
		//Updating grid position
		Vector3 oldGridPos = m_gridOriginPosition;
		m_gridOriginPosition.x += -m_rotationPoint.x*cos(m_gridOrientation) + m_rotationPoint.y*sin(m_gridOrientation);
		m_gridOriginPosition.y += -m_rotationPoint.x*sin(m_gridOrientation) - m_rotationPoint.y*cos(m_gridOrientation);

		m_gridOriginPosition += m_rotationPoint;

		Vector3 tempLocalAxisX = Vector3(1, 0, 0);// - m_rotationPoint;
		m_localAxisX.x = tempLocalAxisX.x*cos(m_gridOrientation) - tempLocalAxisX.y*sin(m_gridOrientation);
		m_localAxisX.y = tempLocalAxisX.x*sin(m_gridOrientation) + tempLocalAxisX.y*cos(m_gridOrientation);
		m_localAxisX.normalize();

		Vector3 tempLocalAxisY = Vector3(0, 1, 0);// - m_rotationPoint;
		m_localAxisY.x = tempLocalAxisY.x*cos(m_gridOrientation) - tempLocalAxisY.y*sin(m_gridOrientation);
		m_localAxisY.y = tempLocalAxisY.x*sin(m_gridOrientation) + tempLocalAxisY.y*cos(m_gridOrientation);
		m_localAxisY.normalize();
	}

	void ParticleSystem3D::updateEmission(Scalar dt) {
		if(m_params.EmitterAnimation == oscillateZ) {
			m_oscillationZ += dt*m_params.oscillationSpeed;
			m_params.EmitterPosition.z = m_EmitterInitialPosition.z + sin(m_oscillationZ)*m_params.oscillationSize;
		}
		if(m_params.EmitterAnimation == oscillateYZ) {
			m_oscillationZ += dt*m_params.oscillationSpeed;
			m_params.EmitterPosition.z = m_EmitterInitialPosition.z + sin(m_oscillationZ)*m_params.oscillationSize;
			m_params.EmitterPosition.y = m_EmitterInitialPosition.y + sin(m_oscillationZ)*m_params.oscillationSize;
		}

		int totalSpawnedParticles = numInitializedParticles;
		for(int i = numInitializedParticles; numInitializedParticles - totalSpawnedParticles < m_params.EmitterSpawnRatio*dt; i++, numInitializedParticles++) {
			if(i >= m_params.numParticles)
				i = 0;

			m_pParticlesPosition[i].x = m_params.EmitterPosition.x + m_params.EmitterSize.x*(rand()/(float) RAND_MAX);
			m_pParticlesPosition[i].y = m_params.EmitterPosition.y + m_params.EmitterSize.y*(rand()/(float) RAND_MAX);
			m_pParticlesPosition[i].z = m_params.EmitterPosition.z + m_params.EmitterSize.z*(rand()/(float) RAND_MAX);
			m_pParticlesVelocity[i] = Vector3(0, 0, 0);
			m_pParticlesLifeTime[i] = m_params.particlesLife + m_params.particlesLifeVariance*(rand()/(float) RAND_MAX);
			m_pParticlesVorticity[i] = 0;
		}
	}
	void ParticleSystem3D::resetParticleSystem() {
		for(int i = 0; i < m_params.numParticles; i++) {
			m_pParticlesPosition[i].x = FLT_MIN;
			m_pParticlesPosition[i].y = FLT_MIN;
			m_pParticlesPosition[i].z = FLT_MIN;
			m_pParticlesVelocity[i] = Vector3(0, 0, 0);
			m_pParticlesLifeTime[i] = FLT_MAX;
			m_pParticlesVorticity[i] = 0;
		}

		if(m_params.EmitterType == squareRandomEmitter) {
			for(int i = 0; i < m_params.initialNumParticles; i++) {
				m_pParticlesPosition[i].x = m_params.EmitterPosition.x + m_params.EmitterSize.x*(rand()/(float) RAND_MAX);
				m_pParticlesPosition[i].y = m_params.EmitterPosition.y + m_params.EmitterSize.y*(rand()/(float) RAND_MAX);
				m_pParticlesPosition[i].z = m_params.EmitterPosition.z + m_params.EmitterSize.z*(rand()/(float) RAND_MAX);
				m_pParticlesVelocity[i] = Vector3(0, 0, 0);
				m_pParticlesLifeTime[i] = m_params.particlesLife + m_params.particlesLifeVariance*(rand()/(float) RAND_MAX);
				m_pParticlesVorticity[i] = 0;
			}
		} else if(m_params.EmitterType == circleRandomEmitter) {
			for(int i = 0; i < m_params.initialNumParticles; i++) {
				m_pParticlesPosition[i].x = m_params.EmitterPosition.x + m_params.EmitterSize.x*(rand()/(float) RAND_MAX);
				m_pParticlesPosition[i].y = m_params.EmitterPosition.y + m_params.EmitterSize.x*(rand()/(float) RAND_MAX);
				m_pParticlesPosition[i].z = m_params.EmitterPosition.z + m_params.EmitterSize.z*(rand()/(float) RAND_MAX);
				m_pParticlesVelocity[i] = Vector3(0, 0, 0);
				m_pParticlesLifeTime[i] = m_params.particlesLife + m_params.particlesLifeVariance*(rand()/(float) RAND_MAX);
				m_pParticlesVorticity[i] = 0;
			}
		}
		numInitializedParticles = m_params.initialNumParticles;
	}

	Vector3 ParticleSystem3D::jetShading(int i, Scalar scalarFieldValue, Scalar minValue, Scalar maxValue) {
		Vector3 particleColor;
		float totalDist = maxValue - minValue;
		float wavelength = 420 + ((scalarFieldValue - minValue)/totalDist)*360;

		if(wavelength <= 439){
			particleColor.x	= -(wavelength - 440) / (440.0f - 350.0f);
			particleColor.y	= 0.0;
			particleColor.z = 1.0;
		} else if(wavelength <= 489){
			particleColor.x	= 0.0;
			particleColor.y = (wavelength - 440) / (490.0f - 440.0f);
			particleColor.z	= 1.0;
		} else if(wavelength <= 509){
			particleColor.x = 0.0;
			particleColor.y = 1.0;
			particleColor.z = -(wavelength - 510) / (510.0f - 490.0f);
		} else if(wavelength <= 579){ 
			particleColor.x = (wavelength - 510) / (580.0f - 510.0f);
			particleColor.y = 1.0;
			particleColor.z = 0.0;
		} else if(wavelength <= 644){
			particleColor.x = 1.0;
			particleColor.y = -(wavelength - 645) / (645.0f - 580.0f);
			particleColor.z = 0.0;
		} else if(wavelength <= 780){
			particleColor.x = 1.0;
			particleColor.y = 0.0;
			particleColor.x = 0.0;
		} else {
			particleColor.x = 0.0;
			particleColor.y = 0.0;
			particleColor.z = 0.0;
		}

		return particleColor;
	}
	void ParticleSystem3D::updateParticlesColors(Scalar minScalarValue, Scalar maxScalarValue) {
		//#pragma omp parallel for
		for(int i = 0; i < numInitializedParticles; i++) {
			Scalar scalarFieldValue = 0;
			Vector3 particleLocalPosition;
			particleLocalPosition = (m_pParticlesPosition[i] - m_pHexaGrid->getGridOrigin())/m_dx;
			if(particleLocalPosition.x > 0 && particleLocalPosition.x < m_velocityFieldDimensions.x
				&& particleLocalPosition.y > 0 && particleLocalPosition.y < m_velocityFieldDimensions.y) {
				scalarFieldValue = m_pHexaGrid->getGridData3D()->getPressure(particleLocalPosition.x, particleLocalPosition.y, particleLocalPosition.z);
			}
			scalarFieldValue = clamp(scalarFieldValue, minScalarValue, maxScalarValue);
			m_pParticlesColor[i] = jetShading(i, scalarFieldValue, minScalarValue, maxScalarValue);
		}

		unsigned int sizeParticlesColorVBO = m_params.numParticles*sizeof(Vector3);
		glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeParticlesColorVBO, m_pParticlesColor, GL_DYNAMIC_DRAW);

	}


	void ParticleSystem3D::applyTurbulence(Vector3 &velocity, int i) {
		Scalar temp = sin(m_pParticlesLifeTime[i])*1;

		velocity.x += (rand()/RAND_MAX)*0.1 + sin(m_pParticlesLifeTime[i])*0.1;
		velocity.y += (rand()/RAND_MAX)*0.1 + sin(m_pParticlesLifeTime[i])*0.1;
		velocity.z += (rand()/RAND_MAX)*0.1 + sin(m_pParticlesLifeTime[i])*0.1;
	}

	void ParticleSystem3D::update(Scalar dt) {
		m_gridOriginPosition = m_pHexaGrid->getGridOrigin();
		updateEmission(dt);

		Vector3 EmitterTotalSize = m_params.EmitterPosition + m_params.EmitterSize;
		Scalar dx = m_pHexaGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;

		Scalar minColor = 0; //m_pHexaGrid->getFVRenderer()->m_minScalarFieldVal;
		Scalar maxColor = 1; //m_pHexaGrid->getFVRenderer()->m_maxScalarFieldVal;
		
		updateParticlesColors(minColor, maxColor);

		#pragma omp parallel for
		for(int i = 0; i < m_params.numParticles; i++) {
			Vector3 particleLocalPosition, oldParticleLocalPosition;

			particleLocalPosition = m_pParticlesPosition[i] - m_gridOriginPosition;
			oldParticleLocalPosition = particleLocalPosition;

			if(m_pParticlesPosition[i].x > m_params.particlesMaxBounds.x || 
				m_pParticlesPosition[i].x < m_params.particlesMinBounds.x ||
				m_pParticlesPosition[i].y > m_params.particlesMaxBounds.y || 
				m_pParticlesPosition[i].y < m_params.particlesMinBounds.y ||
				m_pParticlesPosition[i].z < m_params.particlesMinBounds.z ||
				m_pParticlesPosition[i].z > m_params.particlesMaxBounds.z ||
				m_pParticlesLifeTime[i] < 0) {
					m_pParticlesPosition[i].x = m_params.EmitterPosition.x + m_params.EmitterSize.x*(rand()/(float) RAND_MAX);
					m_pParticlesPosition[i].y = m_params.EmitterPosition.y + m_params.EmitterSize.y*(rand()/(float) RAND_MAX);
					m_pParticlesPosition[i].z = m_params.EmitterPosition.z + m_params.EmitterSize.z*(rand()/(float) RAND_MAX);
					m_pParticlesLifeTime[i] = m_params.particlesLife + m_params.particlesLifeVariance*(rand()/(float) RAND_MAX);

					continue;
			} 

			
			if(particleLocalPosition.x/m_dx < m_velocityFieldDimensions.x - 1 && particleLocalPosition.x/m_dx > 1 &&
						particleLocalPosition.y/m_dx < m_velocityFieldDimensions.y - 1 && particleLocalPosition.y/m_dx > 1 &&
						particleLocalPosition.z/m_dx < m_velocityFieldDimensions.z - 1 && particleLocalPosition.z/m_dx > 1)  {
				
				Vector3 interpVel = Vector3(0, 0, 0);// trilinearInterpolation(particleLocalPosition/m_dx, m_pVelocityField, m_velocityFieldDimensions);
				//m_pParticlesVorticity[i] = interpolateScalar(particleLocalPosition/m_dx, m_pHexaGrid->getGridData()->getVorticityPtr(), m_velocityFieldDimensions);
				Scalar vorticity = m_pParticlesVorticity[i];
				m_pParticlesVorticity[i] *= m_dx;

				m_pParticlesVelocity[i] = interpVel;

			} else { //Damp the velocity
				Vector3 vorticity(0, 0, -m_pParticlesVorticity[i]);

				Vector3 transformedVelocity;
				transformedVelocity.x = m_pParticlesVelocity[i].x;
				transformedVelocity.y = m_pParticlesVelocity[i].y;
				transformedVelocity.z = 0;
				transformedVelocity += transformedVelocity.cross(vorticity);

				/*m_pParticlesVelocity[i].x = transformedVelocity.x;
				m_pParticlesVelocity[i].y = transformedVelocity.y;*/
				m_pParticlesVorticity[i] *= m_params.naturalDamping; 
				m_pParticlesVelocity[i] *= m_params.naturalDamping; //- (rand()/(float) RAND_MAX)*0.1;
			}

			

			Vector3 tempPos = m_pParticlesPosition[i] + m_pParticlesVelocity[i]*dt*0.5;
			particleLocalPosition = tempPos - m_gridOriginPosition;
			Vector3 interpVel =  Vector3(0, 0, 0); //trilinearInterpolation(particleLocalPosition/m_dx, m_pVelocityField, m_velocityFieldDimensions);
			applyTurbulence(interpVel, i);
			m_pParticlesPosition[i] += interpVel*dt;
			m_pParticlesLifeTime[i] -= dt;
			
		}
	}

	void ParticleSystem3D::drawLocalAxis() {
		glPushMatrix();
		glLoadIdentity();

		glLineWidth(2.0f);
		glColor3f(0.0f, 1.0f, 0.0f);
		glBegin(GL_LINES);
		glVertex2f(m_gridOriginPosition.x, m_gridOriginPosition.y);
		glVertex2f(m_gridOriginPosition.x + m_localAxisX.x, m_gridOriginPosition.y + m_localAxisX.y);
		glEnd();

		glColor3f(1.0f, 0.0f, 0.0f);
		glBegin(GL_LINES);
		glVertex2f(m_gridOriginPosition.x, m_gridOriginPosition.y);
		glVertex2f(m_gridOriginPosition.x + m_localAxisY.x, m_gridOriginPosition.y + m_localAxisY.y);
		glEnd();

		glPopMatrix();
	}
	void ParticleSystem3D::draw() {
		glDisable(GL_TEXTURE_2D);
		glDisable(GL_LIGHTING);
		glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesVBO);
		glBufferData(GL_ARRAY_BUFFER, m_params.numParticles*sizeof(Vector3), m_pParticlesPosition, GL_DYNAMIC_DRAW);
		glPointSize(5.0f);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);
		glBindBuffer(GL_ARRAY_BUFFER, *m_pParticlesColorVBO);
		glColorPointer(3, GL_FLOAT, 0, (void *) 0);
		glDrawArrays(GL_POINTS, 0, m_params.numParticles);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
		glEnable(GL_TEXTURE_2D);
		glEnable(GL_LIGHTING);
	}
}