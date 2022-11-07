
#include "Loaders/XMLParamsLoader.h"

namespace Chimera {
	namespace Loaders {
		#pragma region LoadingFunctions
		bool XMLParamsLoader::loadTrueOrFalse(TiXmlElement *pNode) {
			string trueOrFalseStr = pNode->GetText();
			transform(trueOrFalseStr.begin(), trueOrFalseStr.end(), trueOrFalseStr.begin(), ::tolower);
			if (trueOrFalseStr == "true") {
				return true;
			}
			else if (trueOrFalseStr == "false") {
				return false;
			}
			return false;
		}
		plataform_t XMLParamsLoader::loadPlatform(TiXmlElement *pPlatformNode) {
			string platformStr = pPlatformNode->GetText();
			transform(platformStr.begin(), platformStr.end(), platformStr.begin(), ::tolower);
			if (platformStr == "gpu")
				return PlataformGPU;
			else if (platformStr == "cpu")
				return PlataformCPU;
			return PlataformCPU;
		}

		gridArrangement_t XMLParamsLoader::loadGridArrangement(TiXmlElement *pGridArrangementNode) {
			string gridArrangement = pGridArrangementNode->GetText();
			transform(gridArrangement.begin(), gridArrangement.end(), gridArrangement.begin(), ::tolower);
			if (gridArrangement == "staggered") {
				return staggeredArrangement;
			}
			else if (gridArrangement == "nodebased" || gridArrangement == "nodalbased" || gridArrangement == "nodal") {
				return nodalArrangement;
			}
			return staggeredArrangement;
		}

		kernelTypes_t XMLParamsLoader::loadKernel(TiXmlElement *pKernelNode) {
			string kernelType = pKernelNode->GetText();
			transform(kernelType.begin(), kernelType.end(), kernelType.begin(), ::tolower);
			if (kernelType == "sph") {
				return SPHkernel;
			}
			else if (kernelType == "bilinear") {
				return bilinearKernel;
			}
			else if (kernelType == "inversedistance") {
				return inverseDistance;
			}
			return SPHkernel;
		}

		particlesSampling_t XMLParamsLoader::loadSampler(TiXmlElement *pSamplerNode) {
			string samplerType = pSamplerNode->GetText();
			transform(samplerType.begin(), samplerType.end(), samplerType.begin(), ::tolower);
			if (samplerType == "stratified" || samplerType == "stratifiedampler") {
				return stratifiedSampling;
			}
			else if (samplerType == "poisson" || samplerType == "poissonsampler" || samplerType == "poissondisc") {
				return poissonSampling;
			}
			return stratifiedSampling;
		}
		
		collisionDetectionMethod_t XMLParamsLoader::loadCollisionDetectionMethod(TiXmlElement *pCollisionDetectionMethod) {
			string collisionDetectionMethod = pCollisionDetectionMethod->GetText();
			transform(collisionDetectionMethod.begin(), collisionDetectionMethod.end(), collisionDetectionMethod.begin(), ::tolower);
			if (collisionDetectionMethod == "ccdbrochu") {
				return continuousCollisionBrochu;
			}
			else if (collisionDetectionMethod == "ccdwang") {
				return continuousCollisionWang;
			}
			else if (collisionDetectionMethod == "nocollisiondetection") {
				return noCollisionDetection;
			}
			else if (collisionDetectionMethod == "cgalsegmentintersection") {
				return cgalSegmentIntersection;
			}
		}

		interpolationMethod_t XMLParamsLoader::loadInterpolation(TiXmlElement *pInterpolationNode) {
			if (pInterpolationNode->FirstChildElement("Method")) {
				TiXmlElement *pMethodNode = pInterpolationNode->FirstChildElement("Method");
				if (pMethodNode) {
					string interpolationMethodStr = pMethodNode->GetText();
					transform(interpolationMethodStr.begin(), interpolationMethodStr.end(), interpolationMethodStr.begin(), ::tolower);

					if (interpolationMethodStr == "bilinearNodal" || interpolationMethodStr == "bilinearstaggered") {
						return Linear;
					}
					else if (interpolationMethodStr == "meanvaluecoordinates") {
						return MeanValueCoordinates;
					}
					else if (interpolationMethodStr == "turbulenceinterpolant") {
						return Turbulence;
					}
					else if (interpolationMethodStr == "bilinearstreamfunction") {
						return LinearStreamfunction;
					}
					else if (interpolationMethodStr == "quadraticstreamfunction") {
						return QuadraticStreamfunction;
					}
					else if (interpolationMethodStr == "subdivisionstreamfunction") {
						return SubdivisStreamfunction;
					}
					else if (interpolationMethodStr == "cubicstreamfunction") {
						return CubicStreamfunction;
					}
				}
			}
			return NotDefined;
		}

		integrationMethod_t XMLParamsLoader::loadIntegrationMethodParams(TiXmlElement *pIntegrationNode) {
			TiXmlElement *pTempNode = pIntegrationNode->FirstChildElement();
			string integrationType(pTempNode->Value());
			transform(integrationType.begin(), integrationType.end(), integrationType.begin(), ::tolower);
			if (integrationType == "rungekuttaadaptive") {
				return RungeKutta_Adaptive;
			}
			else if (integrationType == "rungekutta2") {
				return RungeKutta_2;
			}
			else if (integrationType == "rungekutta4") {
				return RungeKutta_4;
			}
			else if (integrationType == "euler") {
				return forwardEuler;
			}
			//Default
			return RungeKutta_2;
		}

		solidBoundaryType_t XMLParamsLoader::loadSolidWallConditions(TiXmlElement *pSolidWallNode) {
			TiXmlElement *pTempNode = pSolidWallNode->FirstChildElement();
			string integrationType(pTempNode->Value());
			transform(integrationType.begin(), integrationType.end(), integrationType.begin(), ::tolower);
			if (integrationType == "freeslip") {
				return Solid_FreeSlip;
			}
			else if (integrationType == "noslip") {
				return Solid_NoSlip;
			}
			return Solid_NoSlip;
		}

		
		template<class VectorType, template <class> class ArrayType>
		typename DataExporter<VectorType, ArrayType>::configParams_t * XMLParamsLoader::loadLoggingParams(TiXmlElement *pLoggingNode) {
			DataExporter<VectorType, ArrayType>::configParams_t * pParams = new DataExporter<VectorType, ArrayType>::configParams_t();
			if (pLoggingNode->FirstChildElement("LogVelocity")) {
				pParams->logVelocity = true;
				pParams->setVelocityFilename(pLoggingNode->FirstChildElement("LogVelocity")->GetText());
			}
			else {
				pParams->logVelocity = false;
			}
			if (pLoggingNode->FirstChildElement("LogPressure")) {
				pParams->logPressure = true;
				pParams->setPressureFilename(pLoggingNode->FirstChildElement("LogPressure")->GetText());
			}
			else {
				pParams->logPressure = false;
			}
			if (pLoggingNode->FirstChildElement("LogDensity")) {
				pParams->logDensity = true;
				pParams->setDensityFilename(pLoggingNode->FirstChildElement("LogDensity")->GetText());
			}
			else {
				pParams->logDensity = false;
			}
			if (pLoggingNode->FirstChildElement("LogCutCells")) {
				pParams->logSpecialCells = true;
				pParams->setCutCellsFilename(pLoggingNode->FirstChildElement("LogCutCells")->GetText());
			}
			else {
				pParams->logDensity = false;
			}
			return pParams;
		}

		Multigrid::solverParams_t * XMLParamsLoader::loadMultigridParams(TiXmlElement *pMultigridNode) {
			Multigrid::solverParams_t *pSolverParams = new Multigrid::solverParams_t();

			if (pMultigridNode->FirstChildElement("Type") != NULL) {
				string multigridType(pMultigridNode->FirstChildElement("Type")->GetText());
				transform(multigridType.begin(), multigridType.end(), multigridType.begin(), tolower);
				if (multigridType == "fmg_sq")
					pSolverParams->multigridType = Multigrid::FMG_SQUARE_GRID;
				else if (multigridType == "fmg")
					pSolverParams->multigridType = Multigrid::FMG;
				else if (multigridType == "yavneh96")
					pSolverParams->multigridType = Multigrid::YAVNEH_96;
				else
					pSolverParams->multigridType = Multigrid::STANDARD;
			}
			else {
				pSolverParams->multigridType = Multigrid::STANDARD;
			}

			if (pMultigridNode->FirstChildElement("Levels") != NULL)
				pSolverParams->numSubgrids = atoi(pMultigridNode->FirstChildElement("Levels")->GetText());

			if (pMultigridNode->FirstChildElement("nCycles") != NULL)
				pSolverParams->nCycles = atoi(pMultigridNode->FirstChildElement("nCycles")->GetText());

			if (pMultigridNode->FirstChildElement("Tolerance")) {
				pSolverParams->tolerance = atof(pMultigridNode->FirstChildElement("Tolerance")->GetText());
			}

			if (pMultigridNode->FirstChildElement("wSor")) {
				pSolverParams->wSor = atof(pMultigridNode->FirstChildElement("wSor")->GetText());
			}

			if (pMultigridNode->FirstChildElement("PreSmooths")) {
				pSolverParams->preSmooths = atof(pMultigridNode->FirstChildElement("PreSmooths")->GetText());
			}

			if (pMultigridNode->FirstChildElement("PostSmooths")) {
				pSolverParams->postSmooths = atof(pMultigridNode->FirstChildElement("PostSmooths")->GetText());
			}

			if (pMultigridNode->FirstChildElement("SolutionSmooths")) {
				pSolverParams->solutionSmooths = atof(pMultigridNode->FirstChildElement("SolutionSmooths")->GetText());
			}

			if (pMultigridNode->FirstChildElement("OperatorCoarsening") != NULL) {
				string coarseningType(pMultigridNode->FirstChildElement("OperatorCoarsening")->GetText());
				transform(coarseningType.begin(), coarseningType.end(), coarseningType.begin(), tolower);
				if (coarseningType == "directinjection") {
					pSolverParams->operatorsCoarseningType = Multigrid::directInjection;
				}
				else if (coarseningType == "rediscretization") {
					pSolverParams->operatorsCoarseningType = Multigrid::rediscretization;
				}
				else if (coarseningType == "garlekin") {
					pSolverParams->operatorsCoarseningType = Multigrid::garlekin;
				}
				else if (coarseningType == "garlekinsimple") {
					pSolverParams->operatorsCoarseningType = Multigrid::garlekinSimple;
				}
				else if (coarseningType == "geometricalaveraging") {
					pSolverParams->operatorsCoarseningType = Multigrid::geomtricAveraging;
				}
			}

			if (pMultigridNode->FirstChildElement("SmoothingScheme")) {
				string smoothingScheme(pMultigridNode->FirstChildElement("SmoothingScheme")->GetText());
				transform(smoothingScheme.begin(), smoothingScheme.end(), smoothingScheme.begin(), tolower);

				if (smoothingScheme == "gaussseidel") {
					pSolverParams->smoothingScheme = Multigrid::gaussSeidel;
				}
				else if (smoothingScheme == "redblackgaussseidel") {
					pSolverParams->smoothingScheme = Multigrid::redBlackGaussSeidel;
				}
				else if (smoothingScheme == "sor") {
					pSolverParams->smoothingScheme = Multigrid::SOR;
				}
				else if (smoothingScheme == "redblacksor") {
					pSolverParams->smoothingScheme = Multigrid::redBlackSOR;
				}
				else if (smoothingScheme == "gaussjacobi") {
					pSolverParams->smoothingScheme = Multigrid::gaussJacobi;
				}
			}

			if (pMultigridNode->FirstChildElement("WeightingScheme")) {
				string weightingScheme(pMultigridNode->FirstChildElement("WeightingScheme")->GetText());
				transform(weightingScheme.begin(), weightingScheme.end(), weightingScheme.begin(), tolower);

				if (weightingScheme == "fullweighting") {
					pSolverParams->weightingScheme = Multigrid::fullWeighting;
				}
				else if (weightingScheme == "halfweighting") {
					pSolverParams->weightingScheme = Multigrid::halfWeighting;
				}
			}

			pSolverParams->solverCategory = Poisson::Relaxation;
			pSolverParams->solverMethod = MultigridMethod;

			return pSolverParams;
		}


		GLRenderer2D::params_t * XMLParamsLoader::loadRendererParams2D(TiXmlElement *pRenderingNode) {
			TiXmlElement *pTempNode;
			GLRenderer2D::params_t *pParams = new GLRenderer2D::params_t();
			if ((pTempNode = pRenderingNode->FirstChildElement("RenderingOptions")) != NULL) {
				/**FlowSolver type and flowSolverParams initialization */
				if (pTempNode->FirstChildElement("VisualizeGrid") != NULL) {
					string visualizeGridStr = pTempNode->FirstChildElement("VisualizeGrid")->GetText();
					transform(visualizeGridStr.begin(), visualizeGridStr.end(), visualizeGridStr.begin(), ::tolower);
					/*if (visualizeGridStr == "true") {
						m_initializeGridVisualization = true;
					}
					else if (visualizeGridStr == "false") {
						m_initializeGridVisualization = false;
					}*/
				}
			}
			return pParams;
		}	

		GLRenderer3D::params_t * XMLParamsLoader::loadRendererParams3D(TiXmlElement *pRenderingNode) {
			TiXmlElement *pTempNode;
			GLRenderer3D::params_t *pParams = new GLRenderer3D::params_t();
			if ((pTempNode = pRenderingNode->FirstChildElement("RenderingOptions")) != NULL) {
				/**FlowSolver type and flowSolverParams initialization */
				if (pTempNode->FirstChildElement("VisualizeGrid") != NULL) {
					string visualizeGridStr = pTempNode->FirstChildElement("VisualizeGrid")->GetText();
					transform(visualizeGridStr.begin(), visualizeGridStr.end(), visualizeGridStr.begin(), ::tolower);
					/*if (visualizeGridStr == "true") {
					m_initializeGridVisualization = true;
					}
					else if (visualizeGridStr == "false") {
					m_initializeGridVisualization = false;
					}*/
				}
			}
			return pParams;
		}
		
		template typename DataExporter<Vector2, Array2D>::configParams_t *  XMLParamsLoader::loadLoggingParams<Vector2, Array2D>(TiXmlElement *pLoggingNode);
		template typename DataExporter<Vector3, Array3D>::configParams_t *  XMLParamsLoader::loadLoggingParams<Vector3, Array3D>(TiXmlElement *pLoggingNode);
		#pragma endregion
	}
}
