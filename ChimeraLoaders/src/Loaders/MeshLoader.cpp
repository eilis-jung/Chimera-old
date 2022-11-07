#include "Loaders/MeshLoader.h"

namespace Chimera {

	namespace Loaders {
		template<class VectorT>
		vector<PolygonalMesh<VectorT> *> MeshLoader::loadPolyMeshes(TiXmlElement *pObjectsNode, const dimensions_t &gridDimensions, Scalar dx) {
			vector<PolygonalMesh<VectorT> *> meshes;
			TiXmlElement *pMeshNode = pObjectsNode->FirstChildElement("Mesh");
			while (pMeshNode != NULL) {
				if (pMeshNode->FirstChildElement("Filename")) {
					string cgalObjMeshFile = "Geometry/3D/";
					string cgalObjShortFile = pMeshNode->FirstChildElement("Filename")->GetText();
					string objectName;
					size_t tempFound = cgalObjShortFile.rfind("/");
					if (tempFound != string::npos) {
						string tempStr = cgalObjShortFile.substr(tempFound, cgalObjShortFile.length() - tempFound);
						objectName = tempStr.substr(1, tempStr.rfind(".") - 1);
					}
					else {
						objectName = cgalObjShortFile.substr(0, cgalObjShortFile.rfind(".") - 1);
					}
					cgalObjMeshFile += cgalObjShortFile;
					VectorT scale(1, 1, 1);
					VectorT position(0, 0, 0);
					if (pMeshNode->FirstChildElement("Scale")) {
						pMeshNode->FirstChildElement("Scale")->QueryFloatAttribute("x", &scale.x);
						pMeshNode->FirstChildElement("Scale")->QueryFloatAttribute("y", &scale.y);
						pMeshNode->FirstChildElement("Scale")->QueryFloatAttribute("z", &scale.z);
					}
					if (pMeshNode->FirstChildElement("position")) {
						pMeshNode->FirstChildElement("position")->QueryFloatAttribute("x", &position.x);
						pMeshNode->FirstChildElement("position")->QueryFloatAttribute("y", &position.y);
						pMeshNode->FirstChildElement("position")->QueryFloatAttribute("z", &position.z);
					}
					objectName += intToStr(meshes.size());

					meshes.push_back(new PolygonalMesh<VectorT>(position, cgalObjMeshFile, gridDimensions, dx));
					meshes.back()->setName(objectName);
				} else {
					throw(exception("loadPolyMeshes: Mesh filename not found inside mesh node!"));
				}

				pMeshNode = pMeshNode->NextSiblingElement("Mesh");
			}
			return meshes;
		}

		template<class VectorT>
		vector<VectorT> MeshLoader::createGearGeometry(TiXmlElement *pGearNode, uint numSubdivis, const VectorT &position) {
			vector<Vector2> linePoints;
			Scalar innerRadius = 0.0f;
			int numberOfDents = 0;
			Scalar dentSmoothing = 0.75;
			Scalar dentSize = 0.07;
			Scalar dentAngleCorrection = 0;
			if (pGearNode->FirstChildElement("Radius")) {
				innerRadius = atof(pGearNode->FirstChildElement("Radius")->GetText());
			}

			if (pGearNode->FirstChildElement("DentSize")) {
				dentSize = atof(pGearNode->FirstChildElement("DentSize")->GetText());
			}

			if (pGearNode->FirstChildElement("AngleCorrection")) {
				dentAngleCorrection = DegreeToRad(atof(pGearNode->FirstChildElement("AngleCorrection")->GetText()));
			}

			if (pGearNode->FirstChildElement("NumberOfDents")) {
				numberOfDents = atoi(pGearNode->FirstChildElement("NumberOfDents")->GetText());
			}

			//Adjusting number of subdivisions
			int numberOfSubdivisPointPerDent = max((int)floor(numSubdivis / numberOfDents), 2);
			numSubdivis = numberOfSubdivisPointPerDent*numberOfDents;

			Scalar angleDx = DegreeToRad(180.0f / numSubdivis);
			Scalar angleBiggerDx = DegreeToRad(180.0f / numberOfDents);

			Scalar angleTemp = (DegreeToRad(180.f) - angleBiggerDx) / 2;
			DoubleScalar angleCorrection = max(DegreeToRad(90.f) - angleTemp, 0.f);
			angleCorrection += dentAngleCorrection;

			for (int i = 0; i < numberOfDents; i++) {
				//Circular part
				for (int j = 0; j < numberOfSubdivisPointPerDent; j++) {
					Vector2 circlePoint;
					circlePoint.x = cos(i*angleBiggerDx * 2 + angleDx*j);
					circlePoint.y = sin(i*angleBiggerDx * 2 + angleDx*j);
					if (j == 0) { //First, create gear dent
						Vector2 gearPoint = circlePoint*dentSize;
						gearPoint.rotate(-angleCorrection);
						linePoints.push_back(circlePoint*innerRadius + gearPoint + position);
						linePoints.push_back(circlePoint*innerRadius + position);
					}
					else if (j == numberOfSubdivisPointPerDent - 1) { //Last, create gear dent
						linePoints.push_back(circlePoint*innerRadius + position);
						Vector2 gearPoint = circlePoint*dentSize;
						gearPoint.rotate(angleCorrection);
						linePoints.push_back(circlePoint*innerRadius + gearPoint + position);
					}
					else {
						linePoints.push_back(circlePoint*innerRadius + position);
					}
				}
			}
			linePoints.push_back(linePoints.front());


			return linePoints;
		}

		template<class VectorT>
		LineMesh<VectorT> * MeshLoader::loadLineMesh(TiXmlElement *pLineMeshNode, const dimensions_t &gridDimensions, Scalar dx) {
			LineMesh<Vector2>::params_t * pLineMeshParams = new LineMesh<VectorT>::params_t();
			if (pLineMeshNode->FirstChildElement("position")) {
				pLineMeshNode->FirstChildElement("position")->QueryFloatAttribute("x", &pLineMeshParams->position.x);
				pLineMeshNode->FirstChildElement("position")->QueryFloatAttribute("y", &pLineMeshParams->position.y);
			}

			TiXmlElement *pGeometryNode = pLineMeshNode->FirstChildElement("Geometry");
			if (pGeometryNode == nullptr) {
				throw(exception("MeshLoader::loadLineMesh: missing geometry node!"));
			}

			if (pGeometryNode->FirstChildElement("ExtrudeAlongNormalsSize")) {
				Scalar extrudeSize = atof(pGeometryNode->FirstChildElement("ExtrudeAlongNormalsSize")->GetText());
				pLineMeshParams->extrudeAlongNormalWidth = extrudeSize;
			}
			if (pGeometryNode->FirstChildElement("File")) {
				string lineStr = "Geometry/2D/";
				lineStr += pGeometryNode->FirstChildElement("File")->GetText();
				Line<Vector2> *pLine = new Line<Vector2>(pLineMeshParams->position, lineStr);
				pLineMeshParams->initialPoints = pLine->getPoints();

				if (pGeometryNode->FirstChildElement("ClosedMesh")) {
					string closedMeshStr = pGeometryNode->FirstChildElement("ClosedMesh")->GetText();
				}
			}
			else {
				Scalar lengthSize;
				if (pGeometryNode->FirstChildElement("lengthSize")) {
					lengthSize = atof(pGeometryNode->FirstChildElement("lengthSize")->GetText());
				}
				int numSubdivis;
				if (pGeometryNode->FirstChildElement("numSubdivisions")) {
					numSubdivis = atoi(pGeometryNode->FirstChildElement("numSubdivisions")->GetText());
				}

				if (pGeometryNode->FirstChildElement("SinFunction")) {
					Scalar amplitude, frequency;
					pGeometryNode->FirstChildElement("SinFunction")->QueryFloatAttribute("amplitude", &amplitude);
					pGeometryNode->FirstChildElement("SinFunction")->QueryFloatAttribute("frequency", &frequency);

					Scalar dx = lengthSize / (numSubdivis - 1);
					for (int i = 0; i < numSubdivis; i++) {
						Vector2 thinObjectPoint;
						thinObjectPoint.x = dx*i - lengthSize*0.5;
						thinObjectPoint.y = sin(dx*i*PI*frequency)*amplitude;
						pLineMeshParams->initialPoints.push_back(thinObjectPoint + pLineMeshParams->position);
					}
				}
				else if (pGeometryNode->FirstChildElement("VerticalLine")) {
					Scalar dx = lengthSize / (numSubdivis - 1);
					for (int i = 0; i < numSubdivis; i++) {
						Vector2 thinObjectPoint;
						thinObjectPoint.x = 0;
						thinObjectPoint.y = dx*i - lengthSize*0.5;
						pLineMeshParams->initialPoints.push_back(thinObjectPoint + pLineMeshParams->position);
					}
				}
				else if (pGeometryNode->FirstChildElement("HorizontalLine")) {
					Scalar dx = lengthSize / (numSubdivis - 1);
					for (int i = 0; i < numSubdivis; i++) {
						Vector2 thinObjectPoint;
						thinObjectPoint.x = dx*i - lengthSize*0.5;
						thinObjectPoint.y = 0;
						pLineMeshParams->initialPoints.push_back(thinObjectPoint + pLineMeshParams->position);
					}
				}
				else if (pGeometryNode->FirstChildElement("CircularLine")) {
					Scalar radius = 0.0f;
					if (pGeometryNode->FirstChildElement("CircularLine")->FirstChildElement("Radius")) {
						radius = atof(pGeometryNode->FirstChildElement("CircularLine")->FirstChildElement("Radius")->GetText());
					}
					Scalar dx = 2.0f / (numSubdivis - 1);
					for (int i = 0; i < numSubdivis; i++) {
						Vector2 thinObjectPoint;
						thinObjectPoint.x = cos(dx*i*PI)*radius;
						thinObjectPoint.y = sin(dx*i*PI)*radius;
						pLineMeshParams->initialPoints.push_back(thinObjectPoint + pLineMeshParams->position);
					}
					if (pLineMeshParams->initialPoints.back() != pLineMeshParams->initialPoints.front()) {
						pLineMeshParams->initialPoints.push_back(pLineMeshParams->initialPoints.front());
					}
				}
				else if (pGeometryNode->FirstChildElement("GearLine")) {
					pLineMeshParams->initialPoints = createGearGeometry(pGeometryNode->FirstChildElement("GearLine"), numSubdivis, pLineMeshParams->position);
				}
				else if (pGeometryNode->FirstChildElement("RectangularLine")) {
					Vector2 rectangleSize(1, 1);

					if (pGeometryNode->FirstChildElement("RectangularLine")->FirstChildElement("size")) {
						pGeometryNode->FirstChildElement("RectangularLine")->FirstChildElement("size")->QueryFloatAttribute("x", &rectangleSize.x);
						pGeometryNode->FirstChildElement("RectangularLine")->FirstChildElement("size")->QueryFloatAttribute("y", &rectangleSize.y);
					}

					//Position is the centroid of the rectangularLine
					pLineMeshParams->initialPoints.push_back(pLineMeshParams->position + Vector2(-rectangleSize.x, -rectangleSize.y)*0.5);
					pLineMeshParams->initialPoints.push_back(pLineMeshParams->position + Vector2(rectangleSize.x, -rectangleSize.y)*0.5);
					pLineMeshParams->initialPoints.push_back(pLineMeshParams->position + Vector2(rectangleSize.x, rectangleSize.y)*0.5);
					pLineMeshParams->initialPoints.push_back(pLineMeshParams->position + Vector2(-rectangleSize.x, rectangleSize.y)*0.5);
					pLineMeshParams->initialPoints.push_back(pLineMeshParams->position + Vector2(-rectangleSize.x, -rectangleSize.y)*0.5);
				}
			}

			return new LineMesh<VectorT>(*pLineMeshParams, gridDimensions, dx);
		}

		template vector<PolygonalMesh<Vector3> *>  MeshLoader::loadPolyMeshes<Vector3>(TiXmlElement *pObjectsNode, const dimensions_t &gridDimensions, Scalar dx);		
		template LineMesh<Vector2> *  MeshLoader::loadLineMesh<Vector2>(TiXmlElement *pObjectsNode, const dimensions_t &gridDimensions, Scalar dx);
		template vector<Vector2>  MeshLoader::createGearGeometry<Vector2>(TiXmlElement *pGearNode, uint numSubdivis, const Vector2 &position);

	}
}