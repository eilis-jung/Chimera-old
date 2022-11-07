#include "Loaders/GridLoader.h"

namespace Chimera {

	namespace Loaders {

		HexaGrid * GridLoader::loadHexaGrid(TiXmlElement *pGridNode) {
			Scalar gridSpacing;
			Vector3 initialBoundary, finalBoundary;
			TiXmlElement *pTempNode;

			if (pTempNode = pGridNode->FirstChildElement("GridFile")) {
				return new HexaGrid(pTempNode->GetText());
			}
			else {
				if (pTempNode = pGridNode->FirstChildElement("InitialPoint")) {
					pTempNode->QueryFloatAttribute("x", &initialBoundary.x);
					pTempNode->QueryFloatAttribute("y", &initialBoundary.y);
					pTempNode->QueryFloatAttribute("z", &initialBoundary.z);
				}
				if (pTempNode = pGridNode->FirstChildElement("FinalPoint")) {
					pTempNode->QueryFloatAttribute("x", &finalBoundary.x);
					pTempNode->QueryFloatAttribute("y", &finalBoundary.y);
					pTempNode->QueryFloatAttribute("z", &finalBoundary.z);
				}
				if (pTempNode = pGridNode->FirstChildElement("Spacing")) {
					gridSpacing = atof(pTempNode->GetText());
				}
				else if (pTempNode = pGridNode->FirstChildElement("Dimensions")) {
					dimensions_t tempDimensions;
					pTempNode->QueryIntAttribute("x", &tempDimensions.x);
					pTempNode->QueryIntAttribute("y", &tempDimensions.y);
					pTempNode->QueryIntAttribute("z", &tempDimensions.z);
					gridSpacing = (finalBoundary.x - initialBoundary.x) / (tempDimensions.x + 2);
				}

				return new HexaGrid(initialBoundary, finalBoundary, gridSpacing);
			}			
		}


		QuadGrid * GridLoader::loadQuadGrid(TiXmlElement *pGridNode) {
			Scalar gridSpacing;
			Vector2 initialBoundary, finalBoundary;
			TiXmlElement *pTempNode;

			if (pTempNode = pGridNode->FirstChildElement("GridFile")) {
				return new QuadGrid(pTempNode->GetText());
			}
			else {
				if (pTempNode = pGridNode->FirstChildElement("InitialPoint")) {
					pTempNode->QueryFloatAttribute("x", &initialBoundary.x);
					pTempNode->QueryFloatAttribute("y", &initialBoundary.y);
				}
				if (pTempNode = pGridNode->FirstChildElement("FinalPoint")) {
					pTempNode->QueryFloatAttribute("x", &finalBoundary.x);
					pTempNode->QueryFloatAttribute("y", &finalBoundary.y);
				}
				if (pTempNode = pGridNode->FirstChildElement("Spacing")) {
					gridSpacing = atof(pTempNode->GetText());
				}
				else if (pTempNode = pGridNode->FirstChildElement("Dimensions")) {
					dimensions_t tempDimensions;
					pTempNode->QueryIntAttribute("x", &tempDimensions.x);
					pTempNode->QueryIntAttribute("y", &tempDimensions.y);
					gridSpacing = (finalBoundary.x - initialBoundary.x) / (tempDimensions.x + 2);
				}

				return new QuadGrid(initialBoundary, finalBoundary, gridSpacing);
			}
		}

		template <>
		void GridLoader::loadSolidCells<Vector3>(TiXmlElement *pSolidCellsNode, GridData<Vector3> *pGridData) {
			//Scalar dx = m_pHexaGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;
			//for (int i = 0; i < m_pHexaGrid->getDimensions().x; i++) {
			//	for (int j = 0; j < m_pHexaGrid->getDimensions().y; j++) {
			//		for (int k = 0; k < m_pHexaGrid->getDimensions().z; k++) {
			//			Grid<Vector3>::gridBounds_t bounds;
			//			bounds.lowerBounds.x = m_pHexaGrid->getGridData3D()->getCenterPoint(i, j, k).x - dx;
			//			//bounds.min.y = m_pHexaGrid->getGridData()->getCenterPoint(i, j, k).y - dx + 0.04;
			//			bounds.lowerBounds.y = m_pHexaGrid->getGridData3D()->getCenterPoint(i, j, k).y - dx;
			//			bounds.lowerBounds.z = m_pHexaGrid->getGridData3D()->getCenterPoint(i, j, k).z - dx;

			//			bounds.upperBounds.x = m_pHexaGrid->getGridData3D()->getCenterPoint(i, j, k).x + dx;
			//			//bounds.max.y = m_pHexaGrid->getGridData()->getCenterPoint(i, j, k).y + dx + 0.04;
			//			bounds.upperBounds.y = m_pHexaGrid->getGridData3D()->getCenterPoint(i, j, k).y + dx;
			//			bounds.upperBounds.z = m_pHexaGrid->getGridData3D()->getCenterPoint(i, j, k).z + dx;
			//		}
			//	}
			//}

			/*TiXmlElement *pTempNode;
			pTempNode = pObjectsNode->FirstChildElement();
			while (pTempNode != NULL) {
				if (string(pTempNode->Value()) == "SphereSolid") {
					Vector3 spherePosition;
					Scalar sphereRadius;
					pTempNode->QueryFloatAttribute("px", &spherePosition.x);
					pTempNode->QueryFloatAttribute("py", &spherePosition.y);
					pTempNode->QueryFloatAttribute("pz", &spherePosition.z);
					pTempNode->QueryFloatAttribute("radius", &sphereRadius);
					m_pHexaGrid->loadSolidCircle(spherePosition, sphereRadius);
				}
				pTempNode = pTempNode->NextSiblingElement();
			}*/
		}

		template<>
		void GridLoader::loadDensityField<Vector3>(TiXmlElement *pDensityNode, GridData<Vector3> *pGridData) {
			Vector3 lowerBound, upperBound;
			Scalar dx = pGridData->getGridSpacing();

			GridData3D *pGridData3D = dynamic_cast<GridData3D *>(pGridData);

			if (pDensityNode->FirstChildElement("Cube")) {
				TiXmlElement *pCubeNode = pDensityNode->FirstChildElement("Cube");
				Vector3 recPosition, recSize;
				pCubeNode->QueryFloatAttribute("px", &recPosition.x);
				pCubeNode->QueryFloatAttribute("py", &recPosition.y);
				pCubeNode->QueryFloatAttribute("pz", &recPosition.z);
				pCubeNode->QueryFloatAttribute("sx", &recSize.x);
				pCubeNode->QueryFloatAttribute("sy", &recSize.y);
				pCubeNode->QueryFloatAttribute("sz", &recSize.z);

				/*FlowSolver<Vector3, Array3D>::scalarFieldMarker_t scalarFieldMarker;
				scalarFieldMarker.position = recPosition;
				scalarFieldMarker.size = recSize;
				scalarFieldMarker.value = 1.0f;
				m_densityMarkers.push_back(scalarFieldMarker);*/

				lowerBound.x = recPosition.x / dx;
				lowerBound.y = recPosition.y / dx;
				lowerBound.z = recPosition.z / dx;
				upperBound.x = lowerBound.x + recSize.x / dx;;
				upperBound.y = lowerBound.y + recSize.y / dx;;
				upperBound.z = lowerBound.z + recSize.z / dx;


				for (int i = lowerBound.x; i < upperBound.x; i++) {
					for (int j = lowerBound.y; j < upperBound.y; j++) {
						for (int k = lowerBound.z; k < upperBound.z; k++) {
							pGridData3D->getDensityBuffer().setValueBothBuffers(1, i, j, k);
						}
					}
				}
			}

		}
	}
}
