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


#ifndef __CHIMERA_FLOW_SOLVER__
#define __CHIMERA_FLOW_SOLVER__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraAdvection.h"
#include "ChimeraPoisson.h"
#include "ChimeraCutCells.h"
#include "ChimeraBoundaryConditions.h"

namespace Chimera {

	
	using namespace Grids;
	using namespace BoundaryConditions;
	using namespace Advection;

	/** Flow Solver class. Used as a base class for all types of structured flow solvers. 
	 ** It defines a solving routine (update) which is the base building block function for all types of flow solvers
	 ** based on projection function*/
	
	namespace Solvers {
		template <class VectorT, template <class> class ArrayType>
		class FlowSolver : public PhysicalObject<VectorT> {
		public:
			#pragma region InternalStructures
			typedef struct velocityImpulse_t {
				VectorT velocity;
				VectorT position;
			};

			typedef struct forcingFunction_t {
				VectorT position;
				VectorT size;
				VectorT strength;
			} forcingFunction_t;

			typedef struct rotationalVelocity_t {
				VectorT center;

				Scalar orientation;
				Scalar minRadius;
				Scalar maxRadius;
				Scalar strenght;

				rotationalVelocity_t() {
					minRadius = maxRadius = strenght = orientation = 0.0f;
				}
			};

			typedef struct torusVelocity_t {
				VectorT position;
				Scalar radius;
				Scalar sectionRadius;
				VectorT upDirection;
				Scalar orientation;
				Scalar strength;

				torusVelocity_t() {
					radius = sectionRadius = orientation = strength = 0.0f;
				}
			};

			typedef struct cutFaceVelocity_t {
				VectorT velocity;
				int faceID;
				faceLocation_t faceLocation;
				torusVelocity_t torusVel;

				cutFaceVelocity_t() {
					faceID = -1;
				}
			};

			/* Rectangular scalar field marker */
			typedef struct scalarFieldMarker_t {
				VectorT position;
				VectorT size;
				Scalar value;

				scalarFieldMarker_t() {
					value = 0;
				}
			} scalarFieldMarker_t;

			typedef struct hotSmokeSource_t {
				VectorT position;
				VectorT velocity;
				Scalar size;

				Scalar densityValue;
				Scalar densityVariation;
				Scalar densityBuoyancyCoefficient;

				Scalar temperatureValue;
				Scalar temperatureVariation;
				Scalar temperatureBuoyancyCoefficient;
			}hotSmokeSource_t;


			typedef struct params_t {
				solverType_t solverType;
				PoissonSolver::params_t * pPoissonSolverParams;
				AdvectionBase::baseParams_t *pAdvectionParams;

				solidBoundaryType_t solidBoundaryType;

				/** Velocity impulses */
				vector<velocityImpulse_t> velocityImpulses;
				/** Rotational velocities*/
				vector<rotationalVelocity_t> rotationalVelocities;
				/** Torus velocities*/
				vector<torusVelocity_t> torusVelocities;
				/** Cut face velocities */
				vector<cutFaceVelocity_t> cutFaceVelocities;
				/** Forcing functions */
				vector<forcingFunction_t> forcingFunctions;
				/** Smoke source */
				vector<hotSmokeSource_t *> smokeSources;

				/** Vorticity confinement force*/
				Scalar vorticityConfinementStrength;

				params_t() {
					solidBoundaryType = Solid_NoSlip;
					vorticityConfinementStrength = 0;
				}

				virtual ~params_t() {

				}
			} params_t;

			
			#pragma endregion

		protected:
			#pragma region ClassMembers
			/** Parameters */
			params_t m_params;

			/** Number of iterations performed by the simulation so far */
			int m_numIterations;

			/** Raw grid pointer */
			StructuredGrid<VectorT> *m_pGrid;

			/** Velocity interpolant. Used for debugging process */
			Interpolant<VectorT, ArrayType, VectorT> *m_pVelocityInterpolant;

			/** Intermediary (after advection) velocity interpolant. Used for debugging process */
			Interpolant<VectorT, ArrayType, VectorT> *m_pAuxVelocityInterpolant;

			/** Density interpolant*/
			Interpolant<Scalar, ArrayType, VectorT> *m_pDensityInterpolant;

			/** Temperature interpolant*/
			Interpolant<Scalar, ArrayType, VectorT> *m_pTemperatureInterpolant;

			/** Boundary conditions **/
			vector<BoundaryCondition<VectorT> *> m_boundaryConditions;

			/** Poisson Matrix */
			PoissonMatrix *m_pPoissonMatrix;

			/** Poisson solver */
			PoissonSolver *m_pPoissonSolver;

			/** Dimensions */
			dimensions_t m_dimensions;

			/** Particle-Based Advection */
			Advection::AdvectionBase *m_pAdvection;

			/* Scalar field markers*/
			vector<scalarFieldMarker_t> m_scalarFieldMarkers;

			/** Timers */
			Core::Timer m_totalSimulationTimer;
			Core::Timer m_advectionTimer;
			Core::Timer m_solvePressureTimer;
			Core::Timer m_projectionTimer;
			Core::Timer m_cutCellGenerationTimer;
			#pragma endregion

			#pragma region InitializationFunctions
			/** Initializes the various parameters of different Poisson solver configurations */
			FORCE_INLINE void initializePoissonSolver() {
				switch (m_params.pPoissonSolverParams->solverMethod) {
					case GPU_CG:
					case CPU_CG:
						initalizeCGSolver();
					break;
					case EigenCG:
						initializeEigenCGSolver();
					break;
				
					case MultigridMethod:
						initializeMultigridSolver();
					break;

					case GaussSeidelMethod:
						initializeGaussSeidelSolver();
					break;
				}
			}

			virtual void initalizeCGSolver();
			virtual void initializeEigenCGSolver();
			virtual void initializeGaussSeidelSolver();
			virtual void initializeMultigridSolver();

			/** All sub-classes must initialize interpolants */
			virtual void initializeInterpolants() = 0;
			virtual AdvectionBase * initializeAdvectionClass();
			#pragma endregion

			#pragma region UpdateFunctions
			virtual void updateVorticity() {};
			virtual void updateKineticEnergy() {};
			virtual void updateFineGridDivergence(Interpolant<VectorT, ArrayType, VectorT> *pInterpolant);

			/** Each subclass must update its own Poisson Matrix accordingly.*/
			virtual void updatePoissonBoundaryConditions() {};
			#pragma endregion

		public:

			#pragma region Constructors
			FlowSolver(const params_t &params, StructuredGrid<VectorT> *pGrid) : 
				PhysicalObject(VectorT(), VectorT(), VectorT()), m_params(params) {
				m_pGrid = pGrid;
				m_maxResidual = 0;
				m_linearSolverIterations = 0;
				m_advectionTime = m_solvePressureTime = m_totalSimulationTime = m_projectionTime = 0;
				m_numIterations = 0;
				m_pPoissonSolver = NULL;
				m_pVelocityInterpolant = m_pAuxVelocityInterpolant = NULL;
				m_pDensityInterpolant = NULL;
			}
			#pragma endregion

			#pragma region AnttweakBarVariables
			/** Simulation Stats*/
			Scalar m_maxResidual;
			int m_linearSolverIterations;

			/** Time scalars*/
			Scalar m_advectionTime;
			Scalar m_solvePressureTime;
			Scalar m_projectionTime;
			Scalar m_cutCellGenerationTime;
			Scalar m_totalSimulationTime;

			/** Divergent estimation */
			Scalar m_meanDivergent;
			Scalar m_totalDivergent;
			#pragma endregion
			
			#pragma region AccessFunctions
			FORCE_INLINE const params_t & getParams() const {
				return m_params;
			}
			FORCE_INLINE params_t & getParams() {
				return m_params;
			}


			FORCE_INLINE Advection::AdvectionBase * getAdvectionClass() {
				return m_pAdvection;
			}

			FORCE_INLINE Interpolant<VectorT, ArrayType, VectorT> * getVelocityInterpolant() {
				return m_pVelocityInterpolant;
			}

			/** Divergent */
			FORCE_INLINE Scalar getTotalDivergent() {
				return m_totalDivergent;
			}

			FORCE_INLINE Scalar getMeanDivergent() {
				return m_meanDivergent;
			}

			/** Grid */
			FORCE_INLINE StructuredGrid<VectorT> * getGrid() const {
				return m_pGrid;
			}

			/** Poisson matrix */
			FORCE_INLINE PoissonMatrix * getPoissonMatrix() const {
				return m_pPoissonMatrix;
			}

			FORCE_INLINE PoissonSolver * getPoissonSolver() const {
				return m_pPoissonSolver;
			}

			FORCE_INLINE int getTotalIterations() const {
				return m_numIterations;
			}


			virtual Scalar getTotalKineticEnergy() const {
				return 0;
			}
			#pragma endregion

			#pragma region SolvingFunctions
			virtual void applyForces(Scalar dt) = 0;
			FORCE_INLINE virtual void solvePressure() {
				const Array<Scalar> *pRhs;
				Array<Scalar> *pPressures;

				if (m_dimensions.z == 0) {
					pRhs = &m_pGrid->getGridData2D()->getDivergentArray();
					pPressures = (Array<Scalar> *)(&m_pGrid->getGridData2D()->getPressureArray());
				}
				else {
					pRhs = &m_pGrid->getGridData3D()->getDivergentArray();
					pPressures = (Array<Scalar> *)(&m_pGrid->getGridData3D()->getPressureArray());
				}
				if (m_params.pPoissonSolverParams->platform == PlataformGPU)
					m_pPoissonSolver->solveGPU(pRhs, pPressures);
				else
					m_pPoissonSolver->solveCPU(pRhs, pPressures);

				m_linearSolverIterations = m_pPoissonSolver->getNumberIterations();
			}
			virtual void divergenceFree(Scalar dt) = 0;
			/** Switcher between methods */
			FORCE_INLINE void project(Scalar dt) {
				divergenceFree(dt);
			}

			virtual void enforceBoundaryConditions();
			#pragma endregion

			#pragma region UpdateFunctions
			FORCE_INLINE void updateTimers() {
				m_advectionTime = m_advectionTimer.secondsElapsed();
				m_solvePressureTime = m_solvePressureTimer.secondsElapsed();
				m_projectionTime = m_projectionTimer.secondsElapsed();
				m_totalSimulationTime = m_totalSimulationTimer.secondsElapsed();
				m_cutCellGenerationTime = m_cutCellGenerationTimer.secondsElapsed();
			}
			/** Updates the current flow solver with the given time step */
			virtual void update(Scalar dt) = 0;
			/** Updates the divergent. Can be used to calculate the mass conservation by evaluating the divergent after the
			projection */
			virtual void updateDivergents(Scalar dt);

			virtual void updatePostProjectionDivergence();
			#pragma endregion

			#pragma region DivergenceUpdateFunctions
			virtual Scalar calculateFinalDivergent(int i, int j) { return 0.0f; }
			virtual Scalar calculateFluxDivergent(int i, int j) { i; j; return 0.0f; }
			virtual Scalar calculateFluxDivergent(int i, int j, int k) { i; j; k; return 0.0f; }
			#pragma endregion

		};
	}
	

}

#endif