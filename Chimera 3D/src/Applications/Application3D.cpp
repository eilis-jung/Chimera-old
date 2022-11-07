//  Copyright (c) 2013, Vinicius Costa Azevedo
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without
//	modification, are permitted provided that the following conditions are met: 
//
//1. Redistributions of source code must retain the above copyright notice, this
//	list of conditions and the following disclaimer. 
//	2. Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation
//	and/or other materials provided with the distribution. 
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
//	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//	The views and conclusions contained in the software and documentation are those
//	of the authors and should not be interpreted as representing official policies, 
//	either expressed or implied, of the FreeBSD Project.

#include "Applications/Application3D.h"

namespace Chimera {

	plataform_t Application3D::loadPlataform(TiXmlElement *pNode) {
		if(pNode->FirstChildElement("Plataform") != NULL) {
			if(string(pNode->FirstChildElement("Plataform")->GetText()) == "GPU")
				return PlataformGPU;
		} 
		return PlataformCPU;
	}

	void Application3D::loadSimulationParams() {
		TiXmlElement *pTempNode;
		try {
			if((pTempNode = m_pMainNode->FirstChildElement("FlowSolverConfig")) != NULL) {
				/**FlowSolver type and flowSolverParams initialization */
				if(pTempNode->FirstChildElement("FlowSolverType") != NULL) {
					string solverType = pTempNode->FirstChildElement("FlowSolverType")->GetText();
					if(solverType == "Regular") {
						m_pFlowSolverParams = new FlowSolverParameters(finiteDifferenceMethod);
					} else if(solverType == "NonRegular") {
						m_pFlowSolverParams = new FlowSolverParameters(finiteVolumeMethod);
					} else if(solverType == "CGP") {
						m_pFlowSolverParams = new FlowSolverParameters(coarseGridProjectionMethod);
					} else if(solverType == "CutCell") {
						m_pFlowSolverParams = new FlowSolverParameters(cutCellMethod);
					}
				}
				/** Convection method */
				if(pTempNode->FirstChildElement("ConvectionMethod") != NULL) {
					loadConvectionMethodParams(pTempNode->FirstChildElement("ConvectionMethod"));
				} 
				/** Interpolation method */
				/*if (pTempNode->FirstChildElement("LinearInterpolationMethod") != NULL) {
					m_pFlowSolverParams->setLinearInterpolationParams(loadLinearInterpolationParams(pTempNode->FirstChildElement("LinearInterpolationMethod")));
				}*/
				/** Pressure method */
				if(pTempNode->FirstChildElement("PressureMethod") != NULL) {
					loadPoissonSolverParams(pTempNode->FirstChildElement("PressureMethod"));
				} else {
					throw exception("PressureMethod node not found.");
				}
				/** Projection method */
				if(pTempNode->FirstChildElement("ProjectionMethod") != NULL) {
					loadProjectionMethodParams(pTempNode->FirstChildElement("ProjectionMethod"));
				} 
				/** Far-field method */
				if(pTempNode->FirstChildElement("FarFieldMethod") != NULL) {
					loadFarFieldParams(pTempNode->FirstChildElement("FarFieldMethod"));
				} 
				/** Integration method */
				if(pTempNode->FirstChildElement("IntegrationMethod") != NULL) {
					loadIntegrationMethodParams(pTempNode->FirstChildElement("IntegrationMethod"));
				}
				/** Solid walls conditions */
				if(pTempNode->FirstChildElement("SolidWallType") != NULL) {
					loadSolidWallConditions(pTempNode->FirstChildElement("SolidWallType"));
				}

				
			} else {
				throw exception("FlowSolverConfig node not found.");
			}
		} catch (exception e) {
			exitProgram(e.what());
		}
	}

	void Application3D::loadRenderingParams() {
		TiXmlElement *pTempNode;
		try {
			if ((pTempNode = m_pMainNode->FirstChildElement("RenderingOptions")) != NULL) {
				/**FlowSolver type and flowSolverParams initialization */
				if (pTempNode->FirstChildElement("VisualizeGrid") != NULL) {
					string visualizeGridStr = pTempNode->FirstChildElement("VisualizeGrid")->GetText();
					transform(visualizeGridStr.begin(), visualizeGridStr.end(), visualizeGridStr.begin(), ::tolower);
					if (visualizeGridStr == "true") {
						m_initializeGridVisualization = true;
					}
					else if(visualizeGridStr == "false") {
						m_initializeGridVisualization = false;
					}
				}
			}
		}
		catch (exception e) {
			exitProgram(e.what());
		}
	}

	void Application3D::loadConvectionMethodParams(TiXmlElement *pConvectionNode) {
		TiXmlElement *pTempNode = pConvectionNode->FirstChildElement();
		string convectionType(pTempNode->Value());
		if(convectionType == "SemiLagrangian") {
			//SemiLagrangian configuration
			if(loadPlataform(pTempNode) == PlataformGPU) {
				m_pFlowSolverParams->setConvectionMethod(GPU_S01SemiLagrangian);
			} else {
				m_pFlowSolverParams->setConvectionMethod(SemiLagrangian);
			}
		} else if(convectionType == "ModifiedMacCormack") {
			if(loadPlataform(pTempNode) == PlataformGPU) {
				m_pFlowSolverParams->setConvectionMethod(GPU_S02ModifiedMacCormack);
			} else {
				m_pFlowSolverParams->setConvectionMethod(MacCormack);
			}
		} else if (convectionType == "ParticleBasedAdvection") {
			PBAdvectionFactory pbaFactory(pTempNode);
			m_pFlowSolverParams->setParticleBasedAdvectionParams(pbaFactory.getParams());
			m_pFlowSolverParams->setConvectionMethod(CPU_ParticleBasedAdvection);
		}
		else if (convectionType == "GridBasedAdvection") {
			m_pFlowSolverParams->setConvectionMethod(CPU_GridBasedAdvection);
			/** Integration method */
			if (pConvectionNode->FirstChildElement("PositionIntegration") != NULL) {
				loadIntegrationMethodParams(pConvectionNode->FirstChildElement("PositionIntegration"));
			}
			
		}
	}

	//LinearInterpolant3D<Vector3>::params_t Application3D::loadLinearInterpolationParams(TiXmlElement *pLinearInterpolationNode) {
	//	LinearInterpolant3D<Vector3>::params_t linearParams;
	//	if (pLinearInterpolationNode->FirstChildElement("UseParticlesCache")) {
	//		string useCache(pLinearInterpolationNode->FirstChildElement("UseParticlesCache")->GetText());
	//		transform(useCache.begin(), useCache.end(), useCache.begin(), ::tolower);
	//		if (useCache == "true") {
	//			linearParams.useParticleCache = true;
	//		}
	//	}
	//	if (pLinearInterpolationNode->FirstChildElement("UseCGALAcceleration")) {
	//		string useCGAL(pLinearInterpolationNode->FirstChildElement("UseCGALAcceleration")->GetText());
	//		transform(useCGAL.begin(), useCGAL.end(), useCGAL.begin(), ::tolower);
	//		if (useCGAL == "true") {
	//			linearParams.useCGALAcceleration = true;
	//		}
	//	}
	//	if (pLinearInterpolationNode->FirstChildElement("PosterioriProjection")) {
	//		string posterioriProj(pLinearInterpolationNode->FirstChildElement("PosterioriProjection")->GetText());
	//		transform(posterioriProj.begin(), posterioriProj.end(), posterioriProj.begin(), ::tolower);
	//		if (posterioriProj == "true") {
	//			linearParams.projectWithFaceNormal = true;
	//		}
	//	}
	//	if (pLinearInterpolationNode->FirstChildElement("TreatCollisions")) {
	//		string treatCollisions(pLinearInterpolationNode->FirstChildElement("TreatCollisions")->GetText());
	//		transform(treatCollisions.begin(), treatCollisions.end(), treatCollisions.begin(), ::tolower);
	//		if (treatCollisions == "true") {
	//			//linearParams.treatCollisions = true;
	//		}
	//	}
	//	if (pLinearInterpolationNode->FirstChildElement("InterpolationType")) {
	//		string interpolationType(pLinearInterpolationNode->FirstChildElement("InterpolationType")->GetText());
	//		transform(interpolationType.begin(), interpolationType.end(), interpolationType.begin(), ::tolower);
	//		if (interpolationType == "sbc") {
	//			linearParams.interpolationMethod = sbcInterpolation;
	//		}
	//		else if (interpolationType == "mvc") {
	//			linearParams.interpolationMethod = mvcInterpolation;
	//		}
	//	}
	//	return linearParams;
	//}

	void Application3D::loadPoissonSolverParams(TiXmlElement *pSolverParamsNode) {
		TiXmlElement *pTempNode = pSolverParamsNode->FirstChildElement();
		string solverType(pTempNode->Value());

		if(solverType == "Multigrid") {
			loadMultigridParams(pTempNode);
		} else if(solverType == "ConjugateGradient") {
			ConjugateGradient::solverParams_t *pSolverParams = new ConjugateGradient::solverParams_t();

			if(pTempNode->FirstChildElement("Tolerance") != NULL) 
				pSolverParams->tolerance = atof(pTempNode->FirstChildElement("Tolerance")->GetText());

			if(pTempNode->FirstChildElement("MaxIterations") != NULL) 
				pSolverParams->maxIterations = atoi(pTempNode->FirstChildElement("MaxIterations")->GetText());

			if(pTempNode->FirstChildElement("Preconditioner") != NULL) {
				string preconditionerName(pTempNode->FirstChildElement("Preconditioner")->GetText());
				if(preconditionerName == "Diagonal")
					pSolverParams->preconditioner = ConjugateGradient::Diagonal;
				else if(preconditionerName == "AINV")
					pSolverParams->preconditioner = ConjugateGradient::AINV;
				else if(preconditionerName == "SmoothedAggregation")
					pSolverParams->preconditioner = ConjugateGradient::SmoothedAggregation;
				else if(preconditionerName == "NoPreconditioner")
					pSolverParams->preconditioner = ConjugateGradient::NoPreconditioner;
			}

			m_pFlowSolverParams->getPressureSolverParams().setSpecificSolverParams(pSolverParams);

			if(pTempNode->FirstChildElement("Plataform") != NULL) {
				string plataform(pTempNode->FirstChildElement("Plataform")->GetText());
				if (plataform == "CPU") {
					m_pFlowSolverParams->getPressureSolverParams().setPressureSolverMethod(CPU_CG);
					pSolverParams->cpuOnly = true;
				}
			} else {
				m_pFlowSolverParams->getPressureSolverParams().setPressureSolverMethod(GPU_CG);
			}
		} else if (solverType == "GaussSeidel") {
			GaussSeidel::solverParams_t *pSolverParams = new GaussSeidel::solverParams_t();

			if(pTempNode->FirstChildElement("Tolerance") != NULL) 
				pSolverParams->tolerance = atof(pTempNode->FirstChildElement("Tolerance")->GetText());

			if(pTempNode->FirstChildElement("MaxIterations") != NULL) 
				pSolverParams->maxIterations = atoi(pTempNode->FirstChildElement("MaxIterations")->GetText());

			m_pFlowSolverParams->getPressureSolverParams().setSpecificSolverParams(pSolverParams);
			m_pFlowSolverParams->getPressureSolverParams().setPressureSolverMethod(GaussSeidelMethod);
		}
	}

	void Application3D::loadProjectionMethodParams(TiXmlElement *pProjectionNode) {
		if(loadPlataform(pProjectionNode) == PlataformCPU) {
			m_pFlowSolverParams->setProjectionMethod(CPU_fractionalStep);
		} else {
			m_pFlowSolverParams->setProjectionMethod(GPU_fractionalStep);
		}
	}

	void Application3D::loadFarFieldParams(TiXmlElement *pFarFieldNode) {
		TiXmlElement *pTempNode = pFarFieldNode->FirstChildElement();
		string farFieldType(pTempNode->Value());
		if(farFieldType == "Standard") {
			m_pFlowSolverParams->setFarFieldMethod(StandardFarfield);
		} else if(farFieldType == "Outflow") {
			m_pFlowSolverParams->setFarFieldMethod(OutflowFarfield);
		}
	}

	void Application3D::loadIntegrationMethodParams(TiXmlElement *pIntegrationNode) {
		TiXmlElement *pTempNode = pIntegrationNode->FirstChildElement();
		string integrationType(pTempNode->Value());
		if(integrationType == "RungeKuttaAdaptive") {
			m_pFlowSolverParams->setIntegrationMethod(RungeKutta_Adaptive);
		} else if(integrationType == "RungeKutta2") {
			m_pFlowSolverParams->setIntegrationMethod(RungeKutta_2);
		} else if(integrationType == "RungeKutta4") { //Deprecated
			m_pFlowSolverParams->setIntegrationMethod(RungeKutta_4); 
		}
		else if (integrationType == "Euler") {
			m_pFlowSolverParams->setIntegrationMethod(forwardEuler);
		}
	}

	void Application3D::loadSolidWallConditions(TiXmlElement *pSolidWallNode) {
		TiXmlElement *pTempNode = pSolidWallNode->FirstChildElement();
		string integrationType(pTempNode->Value());
		if(integrationType == "FreeSlip") {
			m_pFlowSolverParams->setSolidBoundaryType(Solid_FreeSlip);
			TiXmlElement *pMixedNodeInterpolationNode = pTempNode->FirstChildElement("MixNodeInterpolation");
			if (pMixedNodeInterpolationNode) {
				string mixedNodeType(pMixedNodeInterpolationNode->GetText());
				transform(mixedNodeType.begin(), mixedNodeType.end(), mixedNodeType.begin(), ::tolower);
				if (mixedNodeType == "unweighted") {
					m_pFlowSolverParams->setMixedNodeInterpolationType(Unweighted);
				}
				else if (mixedNodeType == "weightedextradimensions") {
					m_pFlowSolverParams->setMixedNodeInterpolationType(WeightedExtraDimensions);
				}
				else if (mixedNodeType == "weightednoextradimensions"){
					m_pFlowSolverParams->setMixedNodeInterpolationType(WeightedNoExtraDimensions);
				}
				else if (mixedNodeType == "facevelocity") {
					m_pFlowSolverParams->setMixedNodeInterpolationType(FaceVelocity);
				}
			}
		} else if(integrationType == "NoSlip") {
			m_pFlowSolverParams->setSolidBoundaryType(Solid_NoSlip);
		} 
	}

	collisionDetectionMethod_t Application3D::loadCollisionDetectionMethod(TiXmlElement *pCollisionDetectionMethod) {
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

	HexaGrid * Application3D::loadGrid(TiXmlElement *pGridNode) {
		Scalar gridSpacing;
		Vector3 initialBoundary, finalBoundary;
		TiXmlElement *pTempNode;
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

	/************************************************************************/
	/* Specific params                                                      */
	/************************************************************************/
	void Application3D::loadMultigridParams(TiXmlElement *pMultigridNode) {
		Multigrid::solverParams_t *pSolverParams = new Multigrid::solverParams_t();

		if(pMultigridNode->FirstChildElement("Type") != NULL) {
			string multigridType(pMultigridNode->FirstChildElement("Type")->GetText());
			transform(multigridType.begin(), multigridType.end(), multigridType.begin(), tolower);
			if(multigridType == "fmg_sq")
				pSolverParams->multigridType = Multigrid::FMG_SQUARE_GRID;
			else if(multigridType == "fmg")
				pSolverParams->multigridType = Multigrid::FMG;
			else if(multigridType == "yavneh96")
				pSolverParams->multigridType = Multigrid::YAVNEH_96;
			else 
				pSolverParams->multigridType = Multigrid::STANDARD;
		} else {
			pSolverParams->multigridType = Multigrid::STANDARD;
		}

		if(pMultigridNode->FirstChildElement("Levels") != NULL) 
			pSolverParams->numSubgrids = atoi(pMultigridNode->FirstChildElement("Levels")->GetText());

		if(pMultigridNode->FirstChildElement("nCycles") != NULL) 
			pSolverParams->nCycles = atoi(pMultigridNode->FirstChildElement("nCycles")->GetText());

		if(pMultigridNode->FirstChildElement("Tolerance")) {
			pSolverParams->tolerance = atof(pMultigridNode->FirstChildElement("Tolerance")->GetText());
		}

		if(pMultigridNode->FirstChildElement("wSor")) {
			pSolverParams->wSor = atof(pMultigridNode->FirstChildElement("wSor")->GetText());
		}

		if(pMultigridNode->FirstChildElement("PreSmooths")) {
			pSolverParams->preSmooths = atof(pMultigridNode->FirstChildElement("PreSmooths")->GetText());
		}

		if(pMultigridNode->FirstChildElement("PostSmooths")) {
			pSolverParams->postSmooths = atof(pMultigridNode->FirstChildElement("PostSmooths")->GetText());
		}

		if(pMultigridNode->FirstChildElement("SolutionSmooths")) {
			pSolverParams->solutionSmooths = atof(pMultigridNode->FirstChildElement("SolutionSmooths")->GetText());
		}

		if(pMultigridNode->FirstChildElement("OperatorCoarsening") != NULL) {
			string coarseningType(pMultigridNode->FirstChildElement("OperatorCoarsening")->GetText());
			transform(coarseningType.begin(), coarseningType.end(), coarseningType.begin(), tolower);
			if(coarseningType == "directinjection") {
				pSolverParams->operatorsCoarseningType = Multigrid::directInjection;
			} else if(coarseningType == "rediscretization") {
				pSolverParams->operatorsCoarseningType = Multigrid::rediscretization;
			} else if(coarseningType == "garlekin") {
				pSolverParams->operatorsCoarseningType = Multigrid::garlekin;
			} else if(coarseningType == "garlekinsimple") {
				pSolverParams->operatorsCoarseningType = Multigrid::garlekinSimple;
			} else if(coarseningType == "geometricalaveraging") {
				pSolverParams->operatorsCoarseningType = Multigrid::geomtricAveraging;
			}
		}

		if(pMultigridNode->FirstChildElement("SmoothingScheme")) {
			string smoothingScheme(pMultigridNode->FirstChildElement("SmoothingScheme")->GetText());
			transform(smoothingScheme.begin(), smoothingScheme.end(), smoothingScheme.begin(), tolower);

			if(smoothingScheme == "gaussseidel") {
				pSolverParams->smoothingScheme = Multigrid::gaussSeidel;
			} else if(smoothingScheme == "redblackgaussseidel") {
				pSolverParams->smoothingScheme = Multigrid::redBlackGaussSeidel;
			} else if(smoothingScheme == "sor") {
				pSolverParams->smoothingScheme = Multigrid::SOR;
			} else if(smoothingScheme == "redblacksor") {
				pSolverParams->smoothingScheme = Multigrid::redBlackSOR;
			} else if(smoothingScheme == "gaussjacobi") {
				pSolverParams->smoothingScheme = Multigrid::gaussJacobi;
			}
		}

		if(pMultigridNode->FirstChildElement("WeightingScheme")) {
			string weightingScheme(pMultigridNode->FirstChildElement("WeightingScheme")->GetText());
			transform(weightingScheme.begin(), weightingScheme.end(), weightingScheme.begin(), tolower);

			if(weightingScheme == "fullweighting") {
				pSolverParams->weightingScheme = Multigrid::fullWeighting;
			} else if(weightingScheme == "halfweighting") {
				pSolverParams->weightingScheme = Multigrid::halfWeighting;
			} 
		}

		m_pFlowSolverParams->getPressureSolverParams().setSpecificSolverParams(pSolverParams);
		m_pFlowSolverParams->getPressureSolverParams().setPressureSolverMethod(MultigridMethod);
	}

	void Application3D::loadLoggingParams(TiXmlElement *pLoggingNode) {
		if(pLoggingNode->FirstChildElement("LogVelocity")) {
			m_dataExporterParams.logVelocity = true;
			m_dataExporterParams.setVelocityFilename(pLoggingNode->FirstChildElement("LogVelocity")->GetText());
		} else {
			m_dataExporterParams.logVelocity = false;
		}
		if(pLoggingNode->FirstChildElement("LogPressure")) {
			m_dataExporterParams.logPressure = true;
			m_dataExporterParams.setPressureFilename(pLoggingNode->FirstChildElement("LogPressure")->GetText());
		} else {
			m_dataExporterParams.logPressure = false;
		}
		if(pLoggingNode->FirstChildElement("LogDensity")) {
			m_dataExporterParams.logDensity = true;
			m_dataExporterParams.setDensityFilename(pLoggingNode->FirstChildElement("LogDensity")->GetText());
		} else {
			m_dataExporterParams.logDensity = false;
		}
		if(pLoggingNode->FirstChildElement("LogCutCells")) {
			m_dataExporterParams.logSpecialCells = true;
			m_dataExporterParams.setCutCellsFilename(pLoggingNode->FirstChildElement("LogCutCells")->GetText());
		} else {
			m_dataExporterParams.logDensity = false;
		}
		m_dataExporterParams.totalSimulatedTime = m_pPhysicsParams->totalSimulationTime;
	}
}