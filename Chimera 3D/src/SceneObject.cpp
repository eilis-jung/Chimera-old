#include "SceneObject.h"
#include "Resources/ResourceManager.h"
#include "Rendering/GLRenderer3D.h"

namespace Chimera {

	/************************************************************************/
	/* ctors                                                                */
	/************************************************************************/

	SceneObject::SceneObject(const string &objectName, const string &objectMaterial) 
		: Object3D(Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(1, 1, 1), GLRenderer3D::getInstance()->getCamera()) {

		Logger::get () << "Loading Scene Object mesh: " << objectName <<  endl;
		m_pMesh = ResourceManager::getInstance()->loadMesh("Models/" + objectName + ".obj");
		m_pMesh->uniformScale(1);
		m_pMesh->generateFlatNormals();
		m_pMesh->updateMeshCenter();
		m_pMesh->dumpBounds();
		//m_pMesh->setMaterial(ResourceManager::getInstance()->loadMaterial("Materials/" + objectMaterial));
	}


}