#ifndef _CHIMERA_SCENE_LOADER_H_
#define _CHIMERA_SCENE_LOADER_H_
#pragma  once

#include "ChimeraCore.h"
#include "ChimeraMesh.h"
#include "ChimeraRendering.h"
#include "ChimeraResources.h"
#include "Rendering/GLRenderer3D.h"

namespace Chimera {
	using namespace Meshes;
	class SceneLoader {


		public:
			/** Loads an XML file scene objects.*/
			SceneLoader(HexaGrid *pHexaGrid, GLRenderer3D *pRenderer);

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			void loadScene(TiXmlElement *pObjectsNode, bool initializePhysics = true);

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			HexaGrid * getHexaGrid() const {
				return m_pHexaGrid;
			}

			void setHexaGrid(HexaGrid* pGrid) {
				m_pHexaGrid = pGrid;
			}

			ParticleSystem3D * getParticleSystem()  {
				return m_pParticleSystem;
			}

		/*	const vector<Plane *> & getPlaneVec() {
				return m_planes;
			}*/

			/*ParticleSystem3D * getParticleSystem()  {
				return m_pParticleSystem;
			}*/

			/*const vector<Mesh *> & getPolygonSurfaces() {
				return m_pPolygonSurfaces;
			}
*/

			
			const vector<Mesh<Vector3, Face> *> & getMeshes() {
				return m_meshes;
			}
	private:
			HexaGrid *m_pHexaGrid;
			bool m_initializePhysics;
			//vector<Plane *> m_planes;
			ParticleSystem3D *m_pParticleSystem;
			vector<Mesh<Vector3, Face> *> m_meshes;
			GLRenderer3D *m_pRenderer;

			/** Scalar field markers */
			vector<FlowSolver<Vector3, Array3D>::scalarFieldMarker_t> m_scalarFieldMarkers;

			//void loadPlane(TiXmlElement *pPlaneNode);
			void loadDensityField(TiXmlElement *pDensityFieldNode);
			void loadParticleSystem(TiXmlElement *pParticleSystemNode);
			ParticleSystem3D::emitter_t loadEmitter(TiXmlElement *pEmitterNode);
			void loadMesh(TiXmlElement *pCgalObjMesh);
			/*PhysicalObject<Vector3>::velocityFunction_t loadVelocityFunction(TiXmlElement *pVelocityFunctionNode);
			PhysicalObject<Vector3>::rotationFunction_t loadRotationFunction(TiXmlElement *pVelocityFunctionNode);*/
	};

}

#endif
