#include "Floor.h"
#include "Resources/ResourceManager.h"

namespace Chimera { 

	Floor::Floor() : Object3D(	Vector3(0, -0.01, 0), Vector3(0, 0, 0), Vector3(1, 1, 1)) {
		Material floorMaterial;     

		floorMaterial.Kd[0] = 0.4f;
		floorMaterial.Kd[1] = 0.2f;
		floorMaterial.Kd[2] = 0.6f;

		m_floorMaterial = ResourceManager::getInstance()->loadMaterial("Materials/Floor.mtl");

	}

	Floor::Floor(const string &materialName, Scalar planeSize /* = 1 */, Scalar tilingAmount /* = 200 */)
		: Object3D( Vector3(0, -0.01, 0), Vector3(0, 0, 0), Vector3(1, 1, 1)) {
		Material floorMaterial;
		m_planeSize = planeSize;
		m_tilingAmount = tilingAmount;
		m_floorMaterial = ResourceManager::getInstance()->loadMaterial("Materials/" + materialName);
	}

	void Floor::draw() {
		glPushMatrix();
			glTranslatef(m_position.x, m_position.y, m_position.z);
			glScalef(m_scale.x, m_scale.y, m_scale.z);
			
			m_floorMaterial->applyMaterial(); 
			glBegin(GL_QUADS);
				glTexCoord2f(0.0f, 0.0f);
				glNormal3f(0, 1, 0);
				glVertex3f(-m_planeSize, 0, -m_planeSize);

				glTexCoord2f(m_tilingAmount, 0);
				glNormal3f(0, 1, 0);
				glVertex3f(-m_planeSize, 0, m_planeSize);

				glTexCoord2f(m_tilingAmount, m_tilingAmount);
				glNormal3f(0, 1, 0);
				glVertex3f(m_planeSize, 0, m_planeSize);
			
				glTexCoord2f(0.0f, m_tilingAmount);
				glNormal3f(0, 1, 0);
				glVertex3f(m_planeSize, 0, -m_planeSize);
			glEnd();
			m_floorMaterial->removeMaterial();
		glPopMatrix();
	}
}
	
