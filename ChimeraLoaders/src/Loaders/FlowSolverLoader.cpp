#include "Loaders/FlowSolverLoader.h"
#include "Loaders/XMLParamsLoader.h"
#include "Loaders/ForcingFunctionsLoader.h"

namespace Chimera {
	namespace Loaders {

		#pragma region LoadingFunctions
		template <class VectorT, template <class> class ArrayType>
		typename FlowSolver<VectorT, ArrayType>::params_t * FlowSolverLoader<VectorT, ArrayType>::loadSimulationParams(TiXmlElement *pSimulationNode) {
			FlowSolver<VectorT, ArrayType>::params_t * pSolverParams = new FlowSolver<VectorT, ArrayType>::params_t();

			/**FlowSolver type and flowSolverParams initialization */
			if (pSimulationNode->FirstChildElement("FlowSolverType") != nullptr) {
				string solverType = pSimulationNode->FirstChildElement("FlowSolverType")->GetText();
				transform(solverType.begin(), solverType.end(), solverType.begin(), ::tolower);
				if (solverType == "regular") {
					pSolverParams->solverType = finiteDifferenceMethod;
				}
				else if (solverType == "cutcell" || solverType == "cutvoxel") {
					pSolverParams->solverType = cutCellMethod;
				}
				else if (solverType == "cutcellso" || solverType == "cutcellsecondorder") {
					pSolverParams->solverType = cutCellSOMethod;
				}
				else if (solverType == "streamfunctionvorticity") {
					pSolverParams->solverType = streamfunctionVorticity;
				}
			}
			else {
				throw exception("FlowSolverType node not found");
			}

			//if (pSimulationNode->FirstChildElement("LogTimeStep") != nullptr) {
			//	int logTimeStep = pSimulationNode->FirstChildElement("LogTimeStep")->QueryFloatAttribute("value", &pParams->totalSimulationTime);
			//	pSolverParams->
			//}

			/** Advection method */
			if (pSimulationNode->FirstChildElement("AdvectionMethod") != NULL) {
				pSolverParams->pAdvectionParams = loadAdvectionParams(pSimulationNode->FirstChildElement("AdvectionMethod"));
			}
			else {
				throw exception("AdvectionMethod node not found.");
			}

			/** Pressure method */
			if (pSimulationNode->FirstChildElement("PressureMethod") != NULL) {
				pSolverParams->pPoissonSolverParams = loadPoissonSolverParams(pSimulationNode->FirstChildElement("PressureMethod"));
			}
			else {
				throw exception("PressureMethod node not found.");
			}
			if (pSolverParams->solverType == cutCellMethod) {
				//Setting up for cutvoxels solver
				pSolverParams->pPoissonSolverParams->solveThinBoundaries = true;
			}

			/** Solid walls conditions */
			if (pSimulationNode->FirstChildElement("SolidWallType") != NULL) {
				pSolverParams->solidBoundaryType = XMLParamsLoader::getInstance()->loadSolidWallConditions(pSimulationNode->FirstChildElement("SolidWallType"));
			}
			else {
				pSolverParams->solidBoundaryType = Solid_NoSlip;
			}

			/** Forcing Functions */
			if (pSimulationNode->FirstChildElement("ForcingFunctions") != NULL) {
				pSolverParams->smokeSources = ForcingFunctionsLoader::getInstance()->loadHotSmokeSources<VectorT, ArrayType>(pSimulationNode->FirstChildElement("ForcingFunctions"));
			}

			TiXmlElement *pVorticityConfinementNode = pSimulationNode->FirstChildElement("VorticityConfinement");
			if (pVorticityConfinementNode) {
				if (pVorticityConfinementNode->QueryFloatAttribute("strength", &pSolverParams->vorticityConfinementStrength) != TIXML_SUCCESS) {
					pSolverParams->vorticityConfinementStrength = 1;
				}
			}

			///** Impulses */
			//if (pSimulationConfig->FirstChildElement("VelocityImpulse")) {
			//	loadVelocityImpulses(pSimulationConfig->FirstChildElement("VelocityImpulse"));
			//}

			//if (pSimulationConfig->FirstChildElement("TorusVelocityField")) {
			//	vector<FlowSolver<Vector3, Array3D>::torusVelocity_t> torusVelocities = loadTorusVelocityField(pSimulationConfig->FirstChildElement("TorusVelocityField"));
			//	for (int i = 0; i < torusVelocities.size(); i++) {
			//		m_pMainSimCfg->getFlowSolver()->addTorusVelocity(torusVelocities[i]);
			//	}
			//}
			//if (pSimulationConfig->FirstChildElement("InternalVelocityField")) {
			//	loadInternalVelocityField(pSimulationConfig->FirstChildElement("InternalVelocityField"));
			//}
			//if (pSimulationConfig->FirstChildElement("CutFaceVelocity")) {
			//	loadCutFaceVelocity(pSimulationConfig->FirstChildElement("CutFaceVelocity"));
			//}

			return pSolverParams;
		}
		
		template <class VectorT, template <class> class ArrayType>
		typename PhysicsCore<VectorT>::params_t * FlowSolverLoader<VectorT, ArrayType>::loadPhysicsCoreParams(TiXmlElement *pSimulationNode) {
			PhysicsCore<VectorT>::params_t * pParams = new PhysicsCore<VectorT>::params_t ();
			if (pSimulationNode->FirstChildElement("TotalTime")) {
				pSimulationNode->FirstChildElement("TotalTime")->QueryFloatAttribute("value", &pParams->totalSimulationTime);
			}
			return pParams;
		}
		#pragma endregion
		
		#pragma region InternalLoadingFunctions
		template <class VectorT, template <class> class ArrayType>
		PoissonSolver::params_t * FlowSolverLoader<VectorT, ArrayType>::loadPoissonSolverParams(TiXmlElement *pSolverParamsNode) {
			PoissonSolver::params_t *pPoissonParams = new PoissonSolver::params_t();

			TiXmlElement *pTempNode = pSolverParamsNode->FirstChildElement();
			string solverType(pTempNode->Value());
			transform(solverType.begin(), solverType.end(), solverType.begin(), ::tolower);

			if (solverType == "conjugategradient") {
				if (pTempNode->FirstChildElement("Tolerance") != NULL)
					pPoissonParams->maxResidual = atof(pTempNode->FirstChildElement("Tolerance")->GetText());

				if (pTempNode->FirstChildElement("MaxIterations") != NULL)
					pPoissonParams->maxIterations = atoi(pTempNode->FirstChildElement("MaxIterations")->GetText());

				/*if (pTempNode->FirstChildElement("Preconditioner") != NULL) {
				string preconditionerName(pTempNode->FirstChildElement("Preconditioner")->GetText());
				if (preconditionerName == "Diagonal")
				pSolverParams->preconditioner = ConjugateGradient::Diagonal;
				else if (preconditionerName == "AINV")
				pSolverParams->preconditioner = ConjugateGradient::AINV;
				else if (preconditionerName == "SmoothedAggregation")
				pSolverParams->preconditioner = ConjugateGradient::SmoothedAggregation;
				else if (preconditionerName == "NoPreconditioner")
				pSolverParams->preconditioner = ConjugateGradient::NoPreconditioner;
				}*/

				if (pTempNode->FirstChildElement("Platform")) {
					pPoissonParams->platform = XMLParamsLoader::getInstance()->loadPlatform(pTempNode->FirstChildElement("Platform"));
					if (pPoissonParams->platform == PlataformCPU) {
						pPoissonParams->solverMethod = CPU_CG;
					}
					else {
						pPoissonParams->solverMethod = GPU_CG;
					}
				}
			}
			else if (solverType == "gaussseidel") {
				if (pTempNode->FirstChildElement("Tolerance") != NULL)
					pPoissonParams->maxResidual = atof(pTempNode->FirstChildElement("Tolerance")->GetText());

				if (pTempNode->FirstChildElement("MaxIterations") != NULL)
					pPoissonParams->maxIterations = atoi(pTempNode->FirstChildElement("MaxIterations")->GetText());

				pPoissonParams->solverMethod = GaussSeidelMethod;
			}

			return pPoissonParams;
		}

		template <class VectorT, template <class> class ArrayType>
		AdvectionBase::baseParams_t * FlowSolverLoader<VectorT, ArrayType>::loadAdvectionParams(TiXmlElement *pAdvectionNode) {
			AdvectionBase::baseParams_t * pAdvParams = nullptr;
			TiXmlElement *pTempNode = pAdvectionNode->FirstChildElement();
			string advectionType(pTempNode->Value());
			transform(advectionType.begin(), advectionType.end(), advectionType.begin(), ::tolower);
			if (advectionType == "particlebasedadvection") {
				ParticleBasedAdvection<VectorT, ArrayType>::params_t *pPBAParams = new ParticleBasedAdvection<VectorT, ArrayType>::params_t();
				pPBAParams->advectionCategory = LagrangianAdvection;
				if (pTempNode->FirstChildElement("PositionIntegration")) {
					pPBAParams->integrationMethod = XMLParamsLoader::getInstance()->loadIntegrationMethodParams(pTempNode->FirstChildElement("PositionIntegration"));

					TiXmlElement *pInterpolationNode = pTempNode->FirstChildElement("PositionIntegration")->FirstChildElement("Interpolant");
					if (pInterpolationNode) {
						pPBAParams->positionIntegrationInterpolation = XMLParamsLoader::getInstance()->loadInterpolation(pInterpolationNode);
					}
					else {
						pPBAParams->positionIntegrationInterpolation = Linear;
					}
				}

				if (pTempNode->FirstChildElement("GridToParticle")) {
					TiXmlElement *pMethodNode = pTempNode->FirstChildElement("GridToParticle")->FirstChildElement("Method");
					if (pMethodNode) {
						string gridToParticleType = pMethodNode->GetText();
						transform(gridToParticleType.begin(), gridToParticleType.end(), gridToParticleType.begin(), ::tolower);
						if (gridToParticleType == "flip") {
							pPBAParams->gridToParticleTransferMethod = ParticleBasedAdvection<VectorT, ArrayType>::params_t::gridToParticle_t::FLIP;
							Scalar previousVal = pPBAParams->mixFLIP;
							if (pMethodNode->QueryFloatAttribute("mixPIC", &pPBAParams->mixFLIP) == TIXML_NO_ATTRIBUTE) {
								pPBAParams->mixFLIP = previousVal;
							}
						}
						else if (gridToParticleType == "pic") {
							pPBAParams->gridToParticleTransferMethod = ParticleBasedAdvection<VectorT, ArrayType>::params_t::gridToParticle_t::PIC;
						}
						else if (gridToParticleType == "apic") {
							pPBAParams->gridToParticleTransferMethod = ParticleBasedAdvection<VectorT, ArrayType>::params_t::gridToParticle_t::APIC;
						}
						else if (gridToParticleType == "rpic") {
							pPBAParams->gridToParticleTransferMethod = ParticleBasedAdvection<VectorT, ArrayType>::params_t::gridToParticle_t::RPIC;
						}
					}
				}

				if (pTempNode->FirstChildElement("ParticleToGrid")) {
					TiXmlElement *pGridNode = pTempNode->FirstChildElement("ParticleToGrid")->FirstChildElement("GridArrangement");
					if (pGridNode) {
						pPBAParams->gridArrangement = XMLParamsLoader::getInstance()->loadGridArrangement(pGridNode);
					}

					TiXmlElement *pKernel = pTempNode->FirstChildElement("ParticleToGrid")->FirstChildElement("Kernel");
					if (pKernel) {
						pPBAParams->kernelType = XMLParamsLoader::getInstance()->loadKernel(pKernel);
					}

					TiXmlElement *pKernelDanglingCell = pTempNode->FirstChildElement("ParticleToGrid")->FirstChildElement("KernelDanglingCells");
					if (pKernelDanglingCell) {
						pPBAParams->kernelDanglingCells = XMLParamsLoader::getInstance()->loadKernel(pKernelDanglingCell);
					}
				}
				if (pTempNode->FirstChildElement("Sampler")) {
					TiXmlElement *pParticelsPerCell = pTempNode->FirstChildElement("Sampler")->FirstChildElement("ParticlesPerCell");
					if (pParticelsPerCell) {
						pPBAParams->particlesPerCell = atoi(pParticelsPerCell->GetText());
					}
					TiXmlElement *pSampler = pTempNode->FirstChildElement("Sampler")->FirstChildElement("Type");
					if (pSampler) {
						pPBAParams->samplingMethod = XMLParamsLoader::getInstance()->loadSampler(pSampler);
					}
					TiXmlElement *pResampleParticles = pTempNode->FirstChildElement("Sampler")->FirstChildElement("ResampleParticles");
					if (pResampleParticles) {
						pPBAParams->resampleParticles = XMLParamsLoader::getInstance()->loadTrueOrFalse(pResampleParticles);
					}
				}

				pAdvParams = pPBAParams;
			}
			else if (advectionType == "gridbasedadvection") {
				pAdvParams = new AdvectionBase::baseParams_t();
				pAdvParams->advectionCategory = EulerianAdvection;
				TiXmlElement *pMethod = pTempNode->FirstChildElement("Method");
				string methodType(pMethod->GetText());
				transform(methodType.begin(), methodType.end(), methodType.begin(), ::tolower);

				if (methodType == "semilagrangian") {
					pAdvParams->gridBasedAdvectionMethod = SemiLagrangian;
				}
				else if (methodType == "modifiedmaccormack" || methodType == "maccormack") {
					pAdvParams->gridBasedAdvectionMethod = MacCormack;
				}
				else if (methodType == "uscip") {
					pAdvParams->gridBasedAdvectionMethod = USCIP;
				}

				/** Integration method */
				if (pTempNode->FirstChildElement("PositionIntegration") != NULL) {
					pAdvParams->integrationMethod = XMLParamsLoader::getInstance()->loadIntegrationMethodParams(pTempNode->FirstChildElement("PositionIntegration"));
				}

			}
			else {
				throw (exception("No supported advection method inside <AdvectionMethod> found!"));
			}
			return pAdvParams;
		}
		#pragma endregion
		
		template class FlowSolverLoader<Vector2, Array2D>;
		template class FlowSolverLoader<Vector3, Array3D>;
	}
}