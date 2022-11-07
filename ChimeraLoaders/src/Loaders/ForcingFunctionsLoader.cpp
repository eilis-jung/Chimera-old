#include "Loaders/ForcingFunctionsLoader.h"
#include "Loaders/XMLParamsLoader.h"

namespace Chimera {

	namespace Loaders {
		#pragma region LoadingFunctions
		template <class VectorT, template <class> class ArrayType>
		typename FlowSolver<VectorT, ArrayType>::hotSmokeSource_t * ForcingFunctionsLoader::loadHotSmokeSource(TiXmlElement *pHotSmokeSourceNode) {
			FlowSolver<VectorT, ArrayType>::hotSmokeSource_t *pHotSmokeSource = new FlowSolver<VectorT, ArrayType>::hotSmokeSource_t();
			TiXmlElement *pPositionNode = pHotSmokeSourceNode->FirstChildElement("Position");
			if (pPositionNode) {
				pHotSmokeSource->position = XMLParamsLoader::getInstance()->loadVectorFromNode<VectorT>(pPositionNode);
			}
			else {
				throw exception("ForcingFunctionsLoader::loadHotSmokeSource: Position node not found!");
			}
			TiXmlElement *pSizeNode = pHotSmokeSourceNode->FirstChildElement("Size");
			if (pSizeNode) {
				pHotSmokeSource->size = XMLParamsLoader::getInstance()->loadScalarFromNode(pSizeNode);
			}
			else {
				throw exception("ForcingFunctionsLoader::loadHotSmokeSource: Size node not found!");
			}

			TiXmlElement *pVelocityNode = pHotSmokeSourceNode->FirstChildElement("Velocity");
			if (pVelocityNode) {
				pHotSmokeSource->velocity = XMLParamsLoader::getInstance()->loadVectorFromNode<VectorT>(pVelocityNode);
			}

			TiXmlElement *pDensityValueNode = pHotSmokeSourceNode->FirstChildElement("DensityValue");
			if (pDensityValueNode) {
				pHotSmokeSource->densityValue = XMLParamsLoader::getInstance()->loadScalarFromNode(pDensityValueNode);
			}
			else {
				pHotSmokeSource->densityValue = 1;
			}

			TiXmlElement *pDensityVariationNode = pHotSmokeSourceNode->FirstChildElement("DensityVariation");
			if (pDensityVariationNode) {
				pHotSmokeSource->densityVariation = XMLParamsLoader::getInstance()->loadScalarFromNode(pDensityVariationNode);
			}
			else {
				pHotSmokeSource->densityVariation = 0;
			}

			TiXmlElement *pDensityCoefficientNode = pHotSmokeSourceNode->FirstChildElement("DensityBuoyancyCoefficient");
			if (pDensityCoefficientNode) {
				pHotSmokeSource->densityBuoyancyCoefficient = XMLParamsLoader::getInstance()->loadScalarFromNode(pDensityCoefficientNode);
			}
			else {
				pHotSmokeSource->densityBuoyancyCoefficient = 6;
			}
			
			TiXmlElement *pTemperatureValueNode = pHotSmokeSourceNode->FirstChildElement("TemperatureValue");
			if (pTemperatureValueNode) {
				pHotSmokeSource->temperatureValue = XMLParamsLoader::getInstance()->loadScalarFromNode(pTemperatureValueNode);
			}
			else {
				pHotSmokeSource->temperatureValue = 1;
			}

			TiXmlElement *pTemperatureVariationNode = pHotSmokeSourceNode->FirstChildElement("TemperatureVariation");
			if (pTemperatureVariationNode) {
				pHotSmokeSource->temperatureVariation = XMLParamsLoader::getInstance()->loadScalarFromNode(pTemperatureVariationNode);
			}
			else {
				pHotSmokeSource->temperatureVariation = 0;
			}

			TiXmlElement *pTemperatureCoefficientNode = pHotSmokeSourceNode->FirstChildElement("TemperatureBuoyancyCoefficient");
			if (pTemperatureCoefficientNode) {
				pHotSmokeSource->temperatureBuoyancyCoefficient = XMLParamsLoader::getInstance()->loadScalarFromNode(pTemperatureCoefficientNode);
			}
			else {
				pHotSmokeSource->temperatureBuoyancyCoefficient = 6;
			}
			return pHotSmokeSource;
		}

		#pragma endregion
		#pragma region FunctionDeclarations
		template typename FlowSolver<Vector2, Array2D>::hotSmokeSource_t * ForcingFunctionsLoader::loadHotSmokeSource<Vector2, Array2D>(TiXmlElement *pHotSmokeSourceNode);
		template typename FlowSolver<Vector3, Array3D>::hotSmokeSource_t * ForcingFunctionsLoader::loadHotSmokeSource<Vector3, Array3D>(TiXmlElement *pHotSmokeSourceNode);
		#pragma endregion
	}
}