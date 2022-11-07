#ifndef _CHIMERA_SCENE_OBJECT_H_
#define _CHIMERA_SCENE_OBJECT_H_
#pragma  once

#include "ChimeraCore.h"
#include "Primitives/Object3D.h"
#include "Resources/Material.h"
#include "Resources/Mesh.h"


namespace Chimera {

	class SceneObject : public Rendering::Object3D {


	public:
		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		SceneObject(const string &objectName, const string &objectMaterial);

	private:
		shared_ptr<Material> m_floorMaterial;

	};

}

#endif
