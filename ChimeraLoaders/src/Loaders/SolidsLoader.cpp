#include "Loaders/SolidsLoader.h"

namespace Chimera {
	namespace Loaders {

		#pragma region SolidsLoadingFunctions
		template <class VectorT>
		RigidObject2D<VectorT> * SolidsLoader::loadRigidObject2D(TiXmlElement *pRigidObjectNode, const dimensions_t &gridDimensions, Scalar dx) {
			RigidObject2D<VectorT> *pRigidObject;
			LineMesh<VectorT> *pLineMesh = MeshLoader::getInstance()->loadLineMesh<VectorT>(pRigidObjectNode, gridDimensions, dx);

			typename PhysicalObject<VectorT>::positionUpdate_t *pPositionUpdate = nullptr;
			typename PhysicalObject<VectorT>::rotationUpdate_t *pRotationUpdate = nullptr;

			TiXmlElement *pPositionUpdateNode = pRigidObjectNode->FirstChildElement("PositionUpdate");
			if (pPositionUpdateNode) {
				pPositionUpdate = loadPositionUpdate<VectorT>(pPositionUpdateNode);
			}

			TiXmlElement *pRotationUpdateNode = pRigidObjectNode->FirstChildElement("RotationUpdate");
			if (pRotationUpdateNode) {
				pRotationUpdate = loadRotationUpdate<VectorT>(pRotationUpdateNode);
			}
			if (pPositionUpdate && pRotationUpdate) {
				return new RigidObject2D<VectorT>(pLineMesh, *pPositionUpdate, *pRotationUpdate);
			}
			else if (pPositionUpdate) {
				return new RigidObject2D<VectorT>(pLineMesh, *pPositionUpdate);
			}
			else if (pRotationUpdate) {
				pPositionUpdate = new PhysicalObject<VectorT>::positionUpdate_t();
				return new RigidObject2D<VectorT>(pLineMesh, *pPositionUpdate, *pRotationUpdate);
			} 

			return new RigidObject2D<VectorT>(pLineMesh);
		}
		#pragma endregion 

		#pragma region LoadingUtils
		template <class VectorT>
		typename PhysicalObject<VectorT>::positionUpdate_t * SolidsLoader::loadPositionUpdate(TiXmlElement *pPositionUpdateNode) {
			typename PhysicalObject<VectorT>::positionUpdate_t *pPositionUpdate = new PhysicalObject<VectorT>::positionUpdate_t();
			if (pPositionUpdateNode->FirstChildElement("SinFunction")) {
				pPositionUpdateNode->FirstChildElement("SinFunction")->QueryFloatAttribute("amplitude", &pPositionUpdate->amplitude);
				pPositionUpdateNode->FirstChildElement("SinFunction")->QueryFloatAttribute("frequency", &pPositionUpdate->frequency);
				pPositionUpdate->positionUpdateType = positionUpdateType_t::sinFunction;
			}
			else if (pPositionUpdateNode->FirstChildElement("CosineFunction")) {
				pPositionUpdateNode->FirstChildElement("CosineFunction")->QueryFloatAttribute("amplitude", &pPositionUpdate->amplitude);
				pPositionUpdateNode->FirstChildElement("CosineFunction")->QueryFloatAttribute("frequency", &pPositionUpdate->frequency);
				pPositionUpdate->positionUpdateType = positionUpdateType_t::cosineFunction;
			}
			else if (pPositionUpdateNode->FirstChildElement("UniformFunction")) {
				pPositionUpdateNode->FirstChildElement("UniformFunction")->QueryFloatAttribute("amplitude", &pPositionUpdate->amplitude);
				pPositionUpdate->positionUpdateType = positionUpdateType_t::uniformFunction;
			}
			else if (pPositionUpdateNode->FirstChildElement("Path")) {
				string lineStr = "Geometry/2D/";
				lineStr += pPositionUpdateNode->FirstChildElement("Path")->FirstChildElement("File")->GetText();
				Line<Vector2> *pLine = new Line<Vector2>(Vector2(0, 0), lineStr);
				pPositionUpdate->positionUpdateType = positionUpdateType_t::pathAnimation;
				pPositionUpdate->pathMesh = pLine->getPoints();
				pPositionUpdateNode->FirstChildElement("Path")->QueryFloatAttribute("amplitude", &pPositionUpdate->amplitude);
				if (pPositionUpdateNode->FirstChildElement("Path")->FirstChildElement("position")) {
					Vector2 position;
					pPositionUpdateNode->FirstChildElement("Path")->FirstChildElement("position")->QueryFloatAttribute("x", &position.x);
					pPositionUpdateNode->FirstChildElement("Path")->FirstChildElement("position")->QueryFloatAttribute("y", &position.y);
					for (int i = 0; i < pPositionUpdate->pathMesh.size(); i++) {
						pPositionUpdate->pathMesh[i] += position;
					}
				}
			}
			if (pPositionUpdateNode->FirstChildElement("Direction")) {
				pPositionUpdateNode->FirstChildElement("Direction")->QueryFloatAttribute("x", &pPositionUpdate->direction.x);
				pPositionUpdateNode->FirstChildElement("Direction")->QueryFloatAttribute("y", &pPositionUpdate->direction.y);
			}
			return pPositionUpdate;
		}

		template <class VectorT>
		typename PhysicalObject<VectorT>::rotationUpdate_t * SolidsLoader::loadRotationUpdate(TiXmlElement *pRotationUpdateNode) {
			typename PhysicalObject<VectorT>::rotationUpdate_t *pRotationUpdate = new PhysicalObject<VectorT>::rotationUpdate_t();
			TiXmlElement *pRotationType = pRotationUpdateNode->FirstChildElement();
			string rotationTypeStr(pRotationType->Value());
			transform(rotationTypeStr.begin(), rotationTypeStr.end(), rotationTypeStr.begin(), ::tolower);
			pRotationUpdate->rotationType = constantRotation;
			if (rotationTypeStr == "alternating" || rotationTypeStr == "alternate") {
				pRotationUpdate->rotationType = alternatingRotation;
				TiXmlElement *pMinAngleNode = pRotationType->FirstChildElement("MinAngle");
				if (pMinAngleNode) {
					pRotationUpdate->minAngle = DegreeToRad(atof(pMinAngleNode->GetText()));
				}
				else {
					throw(exception("SolidsLoader::loadRotationUpdate MinAngle node not found inside alternating rotation!"));
				}
				TiXmlElement *pMaxAngleNode = pRotationType->FirstChildElement("MaxAngle");
				if (pMaxAngleNode) {
					pRotationUpdate->maxAngle = DegreeToRad(atof(pMaxAngleNode->GetText()));
				}
				else {
					throw(exception("SolidsLoader::loadRotationUpdate MaxAngle node not found inside alternating rotation!"));
				}
			}
			if (pRotationUpdateNode->FirstChildElement("InitialAngle")) {
				pRotationUpdate->initialRotation = DegreeToRad(atof(pRotationUpdateNode->FirstChildElement("InitialAngle")->GetText()));
			}
			if (pRotationUpdateNode->FirstChildElement("AngularSpeed")) {
				pRotationUpdate->speed = atof(pRotationUpdateNode->FirstChildElement("AngularSpeed")->GetText());
			}
			if (pRotationUpdateNode->FirstChildElement("AngularAcceleration")) {
				pRotationUpdate->acceleration = atof(pRotationUpdateNode->FirstChildElement("AngularAcceleration")->GetText());
			}

			if (pRotationUpdateNode->FirstChildElement("StartingTime")) {
				pRotationUpdate->startingTime = atof(pRotationUpdateNode->FirstChildElement("StartingTime")->GetText());
			}

			if (pRotationUpdateNode->FirstChildElement("EndingTime")) {
				pRotationUpdate->endingTime = atof(pRotationUpdateNode->FirstChildElement("EndingTime")->GetText());
			}
			return pRotationUpdate;
		}
		#pragma endregion 

		#pragma region FunctionDeclarations
		template RigidObject2D<Vector2> * SolidsLoader::loadRigidObject2D<Vector2>(TiXmlElement *pRigidObjectNode, const dimensions_t &gridDimensions, Scalar dx);
		template typename PhysicalObject<Vector2>::positionUpdate_t * SolidsLoader::loadPositionUpdate<Vector2>(TiXmlElement *pPositionUpdateNode);
		template typename PhysicalObject<Vector2>::rotationUpdate_t * SolidsLoader::loadRotationUpdate<Vector2>(TiXmlElement *pRotationUpdateNode);
		#pragma endregion 
	}
}