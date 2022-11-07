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

#include "Applications/Application2D.h"

namespace Chimera {
	
	plataform_t Application2D::loadPlataform(TiXmlElement *pNode) {
		if(pNode->FirstChildElement("Plataform") != NULL) {
			if(string(pNode->FirstChildElement("Plataform")->GetText()) == "GPU")
				return PlataformGPU;
		} 
		return PlataformCPU;
	}

	void Application2D::loadSimulationParams() {
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
					}
					else if (solverType == "CutCell") {
						m_pFlowSolverParams = new FlowSolverParameters(cutCellMethod);
					}
					else if (solverType == "Raycast"){
						m_pFlowSolverParams = new FlowSolverParameters(raycastMethod);
					} 
					else if (solverType == "StreamfunctionTurbulence") {
						m_pFlowSolverParams = new FlowSolverParameters(streamfunctionTurbulenceMethod);
					} else if(solverType == "SharpLiquids") {
						m_pFlowSolverParams = new FlowSolverParameters(sharpLiquids);
					}
					else if (solverType == "GhostLiquids") {
						m_pFlowSolverParams = new FlowSolverParameters(ghostLiquids);
					}
				}
				/** Convection method */
				if (pTempNode->FirstChildElement("ConvectionMethod") != NULL) {
					loadConvectionMethodParams(pTempNode->FirstChildElement("ConvectionMethod"));
				}
				/** Pressure method */
				if (pTempNode->FirstChildElement("PressureMethod") != NULL) {
					loadPoissonSolverParams(pTempNode->FirstChildElement("PressureMethod"));
				}
				else {
					throw exception("PressureMethod node not found.");
				}
				/** Projection method */
				if (pTempNode->FirstChildElement("ProjectionMethod") != NULL) {
					loadProjectionMethodParams(pTempNode->FirstChildElement("ProjectionMethod"));
				}
				/** Far-field method */
				if (pTempNode->FirstChildElement("FarFieldMethod") != NULL) {
					loadFarFieldParams(pTempNode->FirstChildElement("FarFieldMethod"));
				}
				/** Integration method */
				if (pTempNode->FirstChildElement("IntegrationMethod") != NULL) {
					loadIntegrationMethodParams(pTempNode->FirstChildElement("IntegrationMethod"));
				}
				/** Solid walls conditions */
				if (pTempNode->FirstChildElement("SolidWallType") != NULL) {
					loadSolidWallConditions(pTempNode->FirstChildElement("SolidWallType"));
				}
			}
 else {
	 throw exception("FlowSolverConfig node not found.");
 }
		}
		catch (exception e) {
			exitProgram(e.what());
		}
	}

	void Application2D::loadConvectionMethodParams(TiXmlElement *pConvectionNode) {
		TiXmlElement *pTempNode = pConvectionNode->FirstChildElement();
		string convectionType(pTempNode->Value());
		if (convectionType == "SemiLagrangian") {
			//SemiLagrangian configuration
			if (loadPlataform(pTempNode) == PlataformGPU) {
				m_pFlowSolverParams->setConvectionMethod(GPU_S01SemiLagrangian);
			}
			else {
				m_pFlowSolverParams->setConvectionMethod(SemiLagrangian);
			}
		}
		else if (convectionType == "ModifiedMacCormack") {
			if (loadPlataform(pTempNode) == PlataformGPU) {
				m_pFlowSolverParams->setConvectionMethod(GPU_S02ModifiedMacCormack);
			}
			else {
				m_pFlowSolverParams->setConvectionMethod(MacCormack);
			}
		}
		else if (convectionType == "FLIP") {
			m_pFlowSolverParams->setConvectionMethod(CPU_ParticleBasedAdvection);
			if (pTempNode->FirstChildElement("PositionIntegration")) {
				string positionIntegrationStr = pTempNode->FirstChildElement("PositionIntegration")->GetText();
				if (positionIntegrationStr == "forwardEuler") {
					m_pFlowSolverParams->getFlipParams().positionIntegration = FLIPParams_t::forwardEuler;
				}
				else if (positionIntegrationStr == "rungeKutta2") {
					m_pFlowSolverParams->getFlipParams().positionIntegration = FLIPParams_t::rungeKutta2;
				}
				else if (positionIntegrationStr == "modifiedRungeKutta2") {
					m_pFlowSolverParams->getFlipParams().positionIntegration = FLIPParams_t::modifiedRungeKutta2;
				}
			}
			if (pTempNode->FirstChildElement("ResampleParticlesCloseToMesh")) {
				string resampleParticlesStr = pTempNode->FirstChildElement("ResampleParticlesCloseToMesh")->GetText();
				if (resampleParticlesStr == "true") {
					m_pFlowSolverParams->getFlipParams().resampleParticlesCloseToMesh = true;
				}
				else if (resampleParticlesStr == "false") {
					m_pFlowSolverParams->getFlipParams().resampleParticlesCloseToMesh = false;
				}
			}
			if (pTempNode->FirstChildElement("ResamplingFactor"))
				m_pFlowSolverParams->getFlipParams().closeToMeshResamplingFactor = atof(pTempNode->FirstChildElement("ResamplingFactor")->GetText());

			if (pTempNode->FirstChildElement("VelocityUpdate")) {
				string velocityUpdateStr = pTempNode->FirstChildElement("VelocityUpdate")->GetText();
				if (velocityUpdateStr == "PIC") {
					m_pFlowSolverParams->getFlipParams().velocityUpdate = FLIPParams_t::PIC;
				}
				else if (velocityUpdateStr == "FLIP") {
					m_pFlowSolverParams->getFlipParams().velocityUpdate = FLIPParams_t::FLIP;
				}
			}
			if (pTempNode->FirstChildElement("ParticleDensity")) {
				m_pFlowSolverParams->getFlipParams().particleDensity = atoi(pTempNode->FirstChildElement("ParticleDensity")->GetText());
			}
			if (pTempNode->FirstChildElement("FixParticlesSampling")) {
				TiXmlElement *particleSamplingNode = pTempNode->FirstChildElement("FixParticlesSampling");
				TiXmlElement *pFixParticleSamplingType = particleSamplingNode->FirstChildElement();
				if (pFixParticleSamplingType) {
					string samplingType(pFixParticleSamplingType->Value());
					transform(samplingType.begin(), samplingType.end(), samplingType.begin(), ::tolower);

					if (samplingType == "finegridresampling") {
						m_pFlowSolverParams->getFlipParams().resamplingMethod = FLIPParams_t::fineGridResampling;
						m_pFlowSolverParams->getFlipParams().fineGridSubdivisionFactor = atoi(pFixParticleSamplingType->FirstChildElement("Subdivisions")->GetText());
					}
					else if (samplingType == "cutcellsresampling") {
						m_pFlowSolverParams->getFlipParams().resamplingMethod = FLIPParams_t::cutCellsResampling;
					}
					else {
						m_pFlowSolverParams->getFlipParams().resamplingMethod = FLIPParams_t::noResampling;
					}

					if (particleSamplingNode->FirstChildElement("CellResamplingThreshold")) {
						m_pFlowSolverParams->getFlipParams().cellResamplingFactor = atof(particleSamplingNode->FirstChildElement("CellResamplingThreshold")->GetText());
					}
				}
				TiXmlElement *pResampleInitialDimNode = particleSamplingNode->FirstChildElement("InitialDimensions");
				if (pResampleInitialDimNode) {
					pResampleInitialDimNode->QueryIntAttribute("x", &m_pFlowSolverParams->getFlipParams().resamplingInitialDimensions.x);
					pResampleInitialDimNode->QueryIntAttribute("y", &m_pFlowSolverParams->getFlipParams().resamplingInitialDimensions.y);
					pResampleInitialDimNode->QueryIntAttribute("z", &m_pFlowSolverParams->getFlipParams().resamplingInitialDimensions.z);
				}
				TiXmlElement *pResampleFinalDimNode = particleSamplingNode->FirstChildElement("FinalDimensions");
				if (pResampleFinalDimNode) {
					pResampleFinalDimNode->QueryIntAttribute("x", &m_pFlowSolverParams->getFlipParams().resamplingFinalDimensions.x);
					pResampleFinalDimNode->QueryIntAttribute("y", &m_pFlowSolverParams->getFlipParams().resamplingFinalDimensions.y);
					pResampleFinalDimNode->QueryIntAttribute("z", &m_pFlowSolverParams->getFlipParams().resamplingFinalDimensions.z);
				}
			}
			if (pTempNode->FirstChildElement("ParticleToGridTransferring")) {
				string particleToGridTransfering = pTempNode->FirstChildElement("ParticleToGridTransferring")->GetText();
				if (particleToGridTransfering == "SPH") {
					m_pFlowSolverParams->getFlipParams().velocityTransfer = FLIPParams_t::sphWeights;
				}
				else if (particleToGridTransfering == "bilinear") {
					m_pFlowSolverParams->getFlipParams().velocityTransfer = FLIPParams_t::bilinearWeights;
				}
				else if (particleToGridTransfering == "inverseDistance") {
					m_pFlowSolverParams->getFlipParams().velocityTransfer = FLIPParams_t::inverseDistanceWeights;
				}
				else if (particleToGridTransfering == "meanValue") {
					m_pFlowSolverParams->getFlipParams().velocityTransfer = FLIPParams_t::meanValueWeights;
				}
			}
			if (pTempNode->FirstChildElement("ParticleToGridTransferringDC")) {
				string particleToGridTransfering = pTempNode->FirstChildElement("ParticleToGridTransferringDC")->GetText();
				if (particleToGridTransfering == "SPH") {
					m_pFlowSolverParams->getFlipParams().velocityTransferDC = FLIPParams_t::sphWeights;
				}
				else if (particleToGridTransfering == "bilinear") {
					m_pFlowSolverParams->getFlipParams().velocityTransferDC = FLIPParams_t::bilinearWeights;
				}
				else if (particleToGridTransfering == "inverseDistance") {
					m_pFlowSolverParams->getFlipParams().velocityTransferDC = FLIPParams_t::inverseDistanceWeights;
				}
				else if (particleToGridTransfering == "meanValue") {
					m_pFlowSolverParams->getFlipParams().velocityTransferDC = FLIPParams_t::meanValueWeights;
				}
			}
			if (pTempNode->FirstChildElement("UseStreamfunction")) {
				string useStreamfunction = pTempNode->FirstChildElement("UseStreamfunction")->GetText();
				transform(useStreamfunction.begin(), useStreamfunction.end(), useStreamfunction.begin(), ::tolower);
				if (useStreamfunction == "true") {
					m_pFlowSolverParams->getFlipParams().useStreamFunction = true;
				}
			}
		} 
		else if(convectionType == "ParticleBasedAdvection") {
			PBAdvectionFactory pbaFactory(pTempNode);
			m_pFlowSolverParams->setParticleBasedAdvectionParams(pbaFactory.getParams());
			m_pFlowSolverParams->setConvectionMethod(CPU_ParticleBasedAdvection);
		}
	}
	
	void Application2D::loadParticleBasedParams(TiXmlElement * pParticleBasedNode) {
		m_pFlowSolverParams->setConvectionMethod(CPU_ParticleBasedAdvection);
		if (pParticleBasedNode->FirstChildElement("PositionIntegration")) {
			string positionIntegrationStr = pParticleBasedNode->FirstChildElement("PositionIntegration")->GetText();
			if (positionIntegrationStr == "forwardEuler") {
				m_pFlowSolverParams->getFlipParams().positionIntegration = FLIPParams_t::forwardEuler;
			}
			else if (positionIntegrationStr == "rungeKutta2") {
				m_pFlowSolverParams->getFlipParams().positionIntegration = FLIPParams_t::rungeKutta2;
			}
			else if (positionIntegrationStr == "modifiedRungeKutta2") {
				m_pFlowSolverParams->getFlipParams().positionIntegration = FLIPParams_t::modifiedRungeKutta2;
			}
		}
		if (pParticleBasedNode->FirstChildElement("ResampleParticlesCloseToMesh")) {
			string resampleParticlesStr = pParticleBasedNode->FirstChildElement("ResampleParticlesCloseToMesh")->GetText();
			if (resampleParticlesStr == "true") {
				m_pFlowSolverParams->getFlipParams().resampleParticlesCloseToMesh = true;
			}
			else if (resampleParticlesStr == "false") {
				m_pFlowSolverParams->getFlipParams().resampleParticlesCloseToMesh = false;
			}
		}
		if (pParticleBasedNode->FirstChildElement("ResamplingFactor"))
			m_pFlowSolverParams->getFlipParams().closeToMeshResamplingFactor = atof(pParticleBasedNode->FirstChildElement("ResamplingFactor")->GetText());

		if (pParticleBasedNode->FirstChildElement("VelocityUpdate")) {
			string velocityUpdateStr = pParticleBasedNode->FirstChildElement("VelocityUpdate")->GetText();
			if (velocityUpdateStr == "PIC") {
				m_pFlowSolverParams->getFlipParams().velocityUpdate = FLIPParams_t::PIC;
			}
			else if (velocityUpdateStr == "FLIP") {
				m_pFlowSolverParams->getFlipParams().velocityUpdate = FLIPParams_t::FLIP;
			}
		}
		if (pParticleBasedNode->FirstChildElement("ParticleDensity")) {
			m_pFlowSolverParams->getFlipParams().particleDensity = atoi(pParticleBasedNode->FirstChildElement("ParticleDensity")->GetText());
		}
		if (pParticleBasedNode->FirstChildElement("FixParticlesSampling")) {
			TiXmlElement *particleSamplingNode = pParticleBasedNode->FirstChildElement("FixParticlesSampling");
			TiXmlElement *pFixParticleSamplingType = particleSamplingNode->FirstChildElement();
			if (pFixParticleSamplingType) {
				string samplingType(pFixParticleSamplingType->Value());
				transform(samplingType.begin(), samplingType.end(), samplingType.begin(), ::tolower);

				if (samplingType == "finegridresampling") {
					m_pFlowSolverParams->getFlipParams().resamplingMethod = FLIPParams_t::fineGridResampling;
					m_pFlowSolverParams->getFlipParams().fineGridSubdivisionFactor = atoi(pFixParticleSamplingType->FirstChildElement("Subdivisions")->GetText());
				}
				else if (samplingType == "cutcellsresampling") {
					m_pFlowSolverParams->getFlipParams().resamplingMethod = FLIPParams_t::cutCellsResampling;
				}
				else {
					m_pFlowSolverParams->getFlipParams().resamplingMethod = FLIPParams_t::noResampling;
				}

				if (particleSamplingNode->FirstChildElement("CellResamplingThreshold")) {
					m_pFlowSolverParams->getFlipParams().cellResamplingFactor = atof(particleSamplingNode->FirstChildElement("CellResamplingThreshold")->GetText());
				}
			}
			TiXmlElement *pResampleInitialDimNode = particleSamplingNode->FirstChildElement("InitialDimensions");
			if (pResampleInitialDimNode) {
				pResampleInitialDimNode->QueryIntAttribute("x", &m_pFlowSolverParams->getFlipParams().resamplingInitialDimensions.x);
				pResampleInitialDimNode->QueryIntAttribute("y", &m_pFlowSolverParams->getFlipParams().resamplingInitialDimensions.y);
				pResampleInitialDimNode->QueryIntAttribute("z", &m_pFlowSolverParams->getFlipParams().resamplingInitialDimensions.z);
			}
			TiXmlElement *pResampleFinalDimNode = particleSamplingNode->FirstChildElement("FinalDimensions");
			if (pResampleFinalDimNode) {
				pResampleFinalDimNode->QueryIntAttribute("x", &m_pFlowSolverParams->getFlipParams().resamplingFinalDimensions.x);
				pResampleFinalDimNode->QueryIntAttribute("y", &m_pFlowSolverParams->getFlipParams().resamplingFinalDimensions.y);
				pResampleFinalDimNode->QueryIntAttribute("z", &m_pFlowSolverParams->getFlipParams().resamplingFinalDimensions.z);
			}
		}
		if (pParticleBasedNode->FirstChildElement("ParticleToGridTransferring")) {
			string particleToGridTransfering = pParticleBasedNode->FirstChildElement("ParticleToGridTransferring")->GetText();
			if (particleToGridTransfering == "SPH") {
				m_pFlowSolverParams->getFlipParams().velocityTransfer = FLIPParams_t::sphWeights;
			}
			else if (particleToGridTransfering == "bilinear") {
				m_pFlowSolverParams->getFlipParams().velocityTransfer = FLIPParams_t::bilinearWeights;
			}
			else if (particleToGridTransfering == "inverseDistance") {
				m_pFlowSolverParams->getFlipParams().velocityTransfer = FLIPParams_t::inverseDistanceWeights;
			}
			else if (particleToGridTransfering == "meanValue") {
				m_pFlowSolverParams->getFlipParams().velocityTransfer = FLIPParams_t::meanValueWeights;
			}
		}
		if (pParticleBasedNode->FirstChildElement("ParticleToGridTransferringDC")) {
			string particleToGridTransfering = pParticleBasedNode->FirstChildElement("ParticleToGridTransferringDC")->GetText();
			if (particleToGridTransfering == "SPH") {
				m_pFlowSolverParams->getFlipParams().velocityTransferDC = FLIPParams_t::sphWeights;
			}
			else if (particleToGridTransfering == "bilinear") {
				m_pFlowSolverParams->getFlipParams().velocityTransferDC = FLIPParams_t::bilinearWeights;
			}
			else if (particleToGridTransfering == "inverseDistance") {
				m_pFlowSolverParams->getFlipParams().velocityTransferDC = FLIPParams_t::inverseDistanceWeights;
			}
			else if (particleToGridTransfering == "meanValue") {
				m_pFlowSolverParams->getFlipParams().velocityTransferDC = FLIPParams_t::meanValueWeights;
			}
		}
		if (pParticleBasedNode->FirstChildElement("UseStreamfunction")) {
			string useStreamfunction = pParticleBasedNode->FirstChildElement("UseStreamfunction")->GetText();
			transform(useStreamfunction.begin(), useStreamfunction.end(), useStreamfunction.begin(), ::tolower);
			if (useStreamfunction == "true") {
				m_pFlowSolverParams->getFlipParams().useStreamFunction = true;
			}
		}
	}

	void Application2D::loadPoissonSolverParams(TiXmlElement *pSolverParamsNode) {
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
		} else if (solverType == "EigenConjugateGradient") {
			EigenConjugateGradient::params_t *pSolverParams = new EigenConjugateGradient::params_t();

			if (pTempNode->FirstChildElement("Tolerance") != NULL)
				pSolverParams->tolerance = atof(pTempNode->FirstChildElement("Tolerance")->GetText());

			if (pTempNode->FirstChildElement("MaxIterations") != NULL)
				pSolverParams->maxIterations = atoi(pTempNode->FirstChildElement("MaxIterations")->GetText());

			m_pFlowSolverParams->getPressureSolverParams().setSpecificSolverParams(pSolverParams);
			m_pFlowSolverParams->getPressureSolverParams().setPressureSolverMethod(EigenCG);
		}
	}

	void Application2D::loadProjectionMethodParams(TiXmlElement *pProjectionNode) {
		if(loadPlataform(pProjectionNode) == PlataformCPU) {
			m_pFlowSolverParams->setProjectionMethod(CPU_fractionalStep);
		} else {
			m_pFlowSolverParams->setProjectionMethod(GPU_fractionalStep);
		}
	}

	void Application2D::loadFarFieldParams(TiXmlElement *pFarFieldNode) {
		TiXmlElement *pTempNode = pFarFieldNode->FirstChildElement();
		string farFieldType(pTempNode->Value());
		if(farFieldType == "Standard") {
			m_pFlowSolverParams->setFarFieldMethod(StandardFarfield);
		} else if(farFieldType == "Outflow") {
			m_pFlowSolverParams->setFarFieldMethod(OutflowFarfield);
		}
	}

	void Application2D::loadIntegrationMethodParams(TiXmlElement *pIntegrationNode) {
		TiXmlElement *pTempNode = pIntegrationNode->FirstChildElement();
		string integrationType(pTempNode->Value());
		if(integrationType == "RungeKuttaAdaptive") {
			m_pFlowSolverParams->setIntegrationMethod(RungeKutta_Adaptive);
		} else if(integrationType == "RungeKutta2") {
			m_pFlowSolverParams->setIntegrationMethod(RungeKutta_2);
		} else if(integrationType == "RungeKutta4") { //Deprecated
			m_pFlowSolverParams->setIntegrationMethod(RungeKutta_4); 
		}
	}

	void Application2D::loadSolidWallConditions(TiXmlElement *pSolidWallNode) {
		TiXmlElement *pTempNode = pSolidWallNode->FirstChildElement();
		string solidWallType(pTempNode->Value());
		if(solidWallType == "FreeSlip") {
			m_pFlowSolverParams->setSolidBoundaryType(Solid_FreeSlip);
		} else if(solidWallType == "NoSlip") {
			m_pFlowSolverParams->setSolidBoundaryType(Solid_NoSlip);
		} else if(solidWallType == "Interpolation") {
			m_pFlowSolverParams->setSolidBoundaryType(Solid_Interpolation);
		} else if(solidWallType == "Extrapolate") {
			m_pFlowSolverParams->setSolidBoundaryType(Solid_Extrapolate);
		}
	}

	void Application2D::loadParticleSystem(TiXmlElement *pParticleSystemNode, QuadGrid *pQuadGrid) {
		ParticleSystem2D::configParams_t m_particleSystemParams;
		if(pParticleSystemNode->FirstChildElement("Emitter")) {
			TiXmlElement *pEmitterNode = pParticleSystemNode->FirstChildElement("Emitter");
			if(pEmitterNode->FirstChildElement("Rectangle")) {
				Vector2 emitterInitialPos;
				Vector2 emitterSize;
				pEmitterNode->FirstChildElement("Rectangle")->QueryFloatAttribute("px", &emitterInitialPos.x);
				pEmitterNode->FirstChildElement("Rectangle")->QueryFloatAttribute("py", &emitterInitialPos.y);
				pEmitterNode->FirstChildElement("Rectangle")->QueryFloatAttribute("sx", &emitterSize.x);
				pEmitterNode->FirstChildElement("Rectangle")->QueryFloatAttribute("sy", &emitterSize.y);

				m_particleSystemParams.emitterPosition = emitterInitialPos;
				m_particleSystemParams.emitterSize = emitterSize;
			}
		}
		if(pParticleSystemNode->FirstChildElement("ParticlesProperties")) {
			TiXmlElement *pParticlesProperties = pParticleSystemNode->FirstChildElement("ParticlesProperties");
			if(pParticlesProperties->FirstChildElement("MaxAmount")) {
				m_particleSystemParams.numParticles = atof(pParticlesProperties->FirstChildElement("MaxAmount")->GetText());
			}
			if(pParticlesProperties->FirstChildElement("InitialAmount")) {
				m_particleSystemParams.initialNumParticles = atof(pParticlesProperties->FirstChildElement("InitialAmount")->GetText());
			}
			if(pParticlesProperties->FirstChildElement("SpawnRatio")) {
				m_particleSystemParams.emitterSpawnRatio = atof(pParticlesProperties->FirstChildElement("SpawnRatio")->GetText());
			}
			if(pParticlesProperties->FirstChildElement("LifeTime")) {
				m_particleSystemParams.particlesLife = atof(pParticlesProperties->FirstChildElement("LifeTime")->GetText());
			}
			if(pParticlesProperties->FirstChildElement("LifeVariance")) {
				m_particleSystemParams.particlesLifeVariance = atof(pParticlesProperties->FirstChildElement("LifeVariance")->GetText());
			}
		}
		if(pParticleSystemNode->FirstChildElement("MinBounds")) {
			pParticleSystemNode->FirstChildElement("MinBounds")->QueryFloatAttribute("px", &m_particleSystemParams.particlesMinBounds.x);
			pParticleSystemNode->FirstChildElement("MinBounds")->QueryFloatAttribute("py", &m_particleSystemParams.particlesMinBounds.y);
		}
		if(pParticleSystemNode->FirstChildElement("MaxBounds")) {
			pParticleSystemNode->FirstChildElement("MaxBounds")->QueryFloatAttribute("px", &m_particleSystemParams.particlesMaxBounds.x);
			pParticleSystemNode->FirstChildElement("MaxBounds")->QueryFloatAttribute("py", &m_particleSystemParams.particlesMaxBounds.y);
		}
		if(pQuadGrid == NULL)
			pQuadGrid = dynamic_cast<QuadGrid *>(m_pMainSimCfg->getFlowSolver()->getGrid());

		m_pParticleSystem = new ParticleSystem2D(m_particleSystemParams, pQuadGrid);
		m_pRenderer->setParticleSystem(m_pParticleSystem);

		//Just once
		m_pParticleSystem->setGridOrigin(pQuadGrid->getGridOrigin());

		/*CutCellSolver2D *pCutCellSolver = dynamic_cast<CutCellSolver2D *>(m_pMainSimCfg->getFlowSolver());
		if(pCutCellSolver) {
			if(pCutCellSolver->getSpecialCellsPtr()) {
				m_pParticleSystem->setCutCells2D(pCutCellSolver->getSpecialCellsPtr());
				m_pParticleSystem->setNodeVelocityField(pCutCellSolver->getNodeVelocityArrayPtr());
			}
		} else {
			RegularGridSolver2D *pRegularGridSolver = dynamic_cast<RegularGridSolver2D *>(m_pMainSimCfg->getFlowSolver());
		}*/
	}

	void Application2D::loadDensityField(TiXmlElement *pDensityNode, QuadGrid *pGrid) {
		if(pDensityNode->FirstChildElement("Rectangle")) {
			TiXmlElement *pRecNode = pDensityNode->FirstChildElement("Rectangle");
			Vector2 recPosition, recSize;
			pRecNode->QueryFloatAttribute("px", &recPosition.x);
			pRecNode->QueryFloatAttribute("py", &recPosition.y);
			pRecNode->QueryFloatAttribute("sx", &recSize.x);
			pRecNode->QueryFloatAttribute("sy", &recSize.y);

			int lowerBoundX, lowerBoundY, upperBoundX, upperBoundY;
			if(m_pFlowSolverParams->getDiscretizationMethod() == finiteVolumeMethod) {
				lowerBoundX = recPosition.x;
				lowerBoundY = recPosition.y;
				upperBoundX = lowerBoundX + recSize.x;
				upperBoundY = lowerBoundY + recSize.y;
			} else {
				Scalar dx = pGrid->getGridData2D()->getScaleFactor(0, 0).x;
				lowerBoundX = floor(recPosition.x/dx); 
				lowerBoundY = floor(recPosition.y/dx);
				upperBoundX = lowerBoundX + floor(recSize.x/dx); 
				upperBoundY = lowerBoundY + floor(recSize.y/dx); 
			}

			for(int i = lowerBoundX; i < upperBoundX; i++) {
				for(int j = lowerBoundY; j < upperBoundY; j++) {
					pGrid->getGridData2D()->getDensityBuffer().setValueBothBuffers(1, i, j);
				}
			}	
		}
	}

	vector<FlowSolver<Vector2, Array2D>::rotationalVelocity_t>  Application2D::loadRotationalVelocityField(TiXmlElement *pRotationalVelocityNode) {
		vector<FlowSolver<Vector2, Array2D>::rotationalVelocity_t> rotationalVelocities;
		while (pRotationalVelocityNode) {
			FlowSolver<Vector2, Array2D>::rotationalVelocity_t rotationalVelocity;
			if (pRotationalVelocityNode->FirstChildElement("position")) {
				pRotationalVelocityNode->FirstChildElement("position")->QueryFloatAttribute("x", &rotationalVelocity.center.x);
				pRotationalVelocityNode->FirstChildElement("position")->QueryFloatAttribute("y", &rotationalVelocity.center.y);
			}
			if (pRotationalVelocityNode->FirstChildElement("MinRadius")) {
				rotationalVelocity.minRadius = atof(pRotationalVelocityNode->FirstChildElement("MinRadius")->GetText());
			}
			if (pRotationalVelocityNode->FirstChildElement("MaxRadius")) {
				rotationalVelocity.maxRadius = atof(pRotationalVelocityNode->FirstChildElement("MaxRadius")->GetText());
			}
			if (pRotationalVelocityNode->FirstChildElement("Orientation")) {
				rotationalVelocity.orientation = atof(pRotationalVelocityNode->FirstChildElement("Orientation")->GetText());
			}
			if (pRotationalVelocityNode->FirstChildElement("Strength")) {
				rotationalVelocity.strenght = atof(pRotationalVelocityNode->FirstChildElement("Strength")->GetText());
			}

			rotationalVelocities.push_back(rotationalVelocity);
			pRotationalVelocityNode = pRotationalVelocityNode->NextSiblingElement();
		}
		return rotationalVelocities;
	}

	QuadGrid * Application2D::loadGrid(TiXmlElement *pGridNode) {
		Scalar gridSpacing;
		Vector2 initialBoundary, finalBoundary;
		TiXmlElement *pTempNode;
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
		return new QuadGrid(initialBoundary, finalBoundary, gridSpacing);
	}
	/************************************************************************/
	/* Specific loaders                                                     */
	/************************************************************************/

	void Application2D::loadMultigridParams(TiXmlElement *pMultigridNode) {
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
		m_pFlowSolverParams->getPressureSolverParams().setPressureSolverMethod(Multigrid);
	}

	void Application2D::loadLoggingParams(TiXmlElement *pLoggingNode) {
		if(pLoggingNode->FirstChildElement("Framerate")) {
			m_dataExporterParams.frameRate = atoi(pLoggingNode->FirstChildElement("Framerate")->GetText());
		}
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
		if(pLoggingNode->FirstChildElement("LogThinObject")) {
			m_dataExporterParams.logThinObject = true;
			/*m_pDataLogger->setDensityFilename(pLoggingNode->FirstChildElement("LogThinObject")->GetText());*/
		} else {
			m_dataExporterParams.logThinObject = false;
		}
		if (pLoggingNode->FirstChildElement("LogScreenshot")) {
			m_dataExporterParams.logScreenshot = true;
			m_dataExporterParams.setScreenshotFilename(pLoggingNode->FirstChildElement("LogScreenshot")->GetText());
			m_dataExporterParams.setScreenSize(glutGet(GLUT_SCREEN_WIDTH), glutGet(GLUT_SCREEN_HEIGHT));
		}
		else {
			m_dataExporterParams.logScreenshot = false;
		}
	}

}