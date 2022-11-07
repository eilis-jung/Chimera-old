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

#ifndef __CHIMERA_PARTICLE_BASED_ADVECTION_H_
#define __CHIMERA_PARTICLE_BASED_ADVECTION_H_
#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"
#include "ChimeraCutCells.h"

#include "ParticleBased/ParticlesToGrid.h"
#include "ParticleBased/GridToParticles.h"
#include "Kernels/TransferKernel.h"
#include "Integration/PositionIntegrator.h"
#include "AdvectionBase.h"

namespace Chimera {

	using namespace Core;
	using namespace Grids;
	using namespace Particles;
	using namespace Interpolation;
	using namespace CutCells;

	namespace Advection {

		template <class VectorType, template <class> class ArrayType>
		class ParticleBasedAdvection : public AdvectionBase {

		public:

			#pragma region InternalStructures 
			typedef struct params_t : public AdvectionBase::baseParams_t {

				typedef enum gridToParticle_t {
					PIC,
					FLIP,
					APIC,
					RPIC,
					Turbulent
				} gridToParticle_t;

				integrationMethod_t positionIntegration;
				interpolationMethod_t positionIntegrationInterpolation;

				gridArrangement_t gridArrangement;
				gridToParticle_t gridToParticleTransferMethod;
				Scalar mixFLIP;

				kernelTypes_t kernelType;
				kernelTypes_t kernelDanglingCells;
				Scalar kernelSize;

				collisionDetectionMethod_t collisionDetectionMethod;
								
				particlesSampling_t samplingMethod;
				int particlesPerCell;
				bool resampleParticles;

				params_t() {
					positionIntegration = RungeKutta_2;
					positionIntegrationInterpolation = Linear;
					gridArrangement = staggeredArrangement;
					kernelType = SPHkernel;
					kernelDanglingCells = SPHkernel;

					collisionDetectionMethod = noCollisionDetection;

					particlesPerCell = 8;
					resampleParticles = true; 
					kernelSize = 1;

					mixFLIP = 0.005;
				}
			} params_t;
			#pragma endregion

			#pragma region ConstructorsDestructors
			ParticleBasedAdvection(params_t *pParams, Interpolant<VectorType, ArrayType, VectorType> *pInterpolant, GridData<VectorType> *pGridData) : AdvectionBase(*pParams) {
				m_pParams = pParams;
				m_pVelocityInterpolant = pInterpolant;
				m_pGridData = pGridData;
				m_pParticlesSampler = initializeParticlesSampler();
				m_pParticlesIntegrator = initializeParticlesIntegrator();
				m_pGridToParticles = initializeGridToParticles();
				m_pParticlesToGrid = initializeParticlesToGrid();
				
				m_pParticlesData = m_pParticlesSampler->getParticlesData();
			}
			#pragma endregion ConstructorsDestructors

			#pragma region UpdateFunctions
			virtual void advect(Scalar dt) {
				updatePositions(dt);
				updateGridAttributes();
			}

			virtual void postProjectionUpdate(Scalar dt) {
				updateParticleAttributes();
			}

			/** Updates particles positions */
			virtual void updatePositions(Scalar dt);
			/** Transfers information from particles to grid */
			virtual void updateGridAttributes();
			/** Transfers infomration from grid to particles */
			virtual void updateParticleAttributes();
			#pragma endregion UpdateFunctions

			#pragma region AccessFunctions
			ParticlesData<VectorType> * getParticlesData() {
				return m_pParticlesData;
			}
			/** In order to comply with the rest of the code we need to keep the same access functions.
			  * TODO: Refactor this to simplify these functions. */
			const vector<VectorType> & getParticlesPosition() const {
				return m_pParticlesData->getPositions();
			}
			const vector<VectorType> & getParticlesVelocities() const {
				return m_pParticlesData->getVelocities();
			}
			
			vector<bool> * getResampledParticlesVecPtr() {
				return &m_pParticlesData->getResampledParticles();
			}
			VectorType * getParticlesPositionsPtr() {
				return &m_pParticlesData->getPositions()[0];
			}
			VectorType * getParticlesVelocitiesPtr() {
				return &m_pParticlesData->getVelocities()[0];
			}
			vector<VectorType> * getParticlesVelocitiesVectorPtr() {
				return &m_pParticlesData->getVelocities();
			}
			vector<VectorType> * getParticlesPositionsVectorPtr() {
				return &m_pParticlesData->getPositions();
			}

			vector<int> * getParticlesTagPtr() {
				vector<int> &particlesTags = m_pParticlesData->getIntegerBasedAttribute("tags");
				return &particlesTags;
			}

			PositionIntegrator<VectorType, ArrayType> * getParticleBasedIntegrator() {
				return m_pParticlesIntegrator;
			}

			ParticlesToGrid<VectorType, ArrayType> * getParticlesToGrid() {
				return m_pParticlesToGrid;
			}

			GridToParticles<VectorType, ArrayType> * getGridToParticles() {
				return m_pGridToParticles;
			}

			void setGridToParticles(GridToParticles<VectorType, ArrayType> *pGridToParticles) {
				m_pGridToParticles = pGridToParticles;
			}

			ParticlesSampler<VectorType, ArrayType> * getParticlesSampler() {
				return m_pParticlesSampler;
			}
			/** Attribute addition functions. This function  adds the same attribute in particles data and an accumulator
			  * in particles to grid class. These attributes are linked by name and can be accesses together in the 
			  * advection phase.*/
			bool addVectorBasedAttribute(string attributeName) {
				return m_pParticlesToGrid->addVectorAttribute(attributeName) && 
						m_pParticlesData->addVectorBasedAttribute(attributeName);
			}

			/** Attribute addition functions. This function  adds the same attribute in particles data and an accumulator
			* in particles to grid class. These attributes are linked by name and can be accesses together in the
			* advection phase. An interpolant is needed if one wants to initialize particle-based values. Otherwise the 
			* newly created buffer receive null values for all particles. */
			bool addScalarBasedAttribute(string attributeName, Interpolant <Scalar, ArrayType, VectorType> *pScalarInterpolant = NULL);

			/** Attribute addition functions. Same as standard ScalarBasedAttribute addition, but it support a custom grid 
			  * size for the accumulators (either coarser, same or finer than the underlying regular grid) */
			bool addScalarBasedAttribute(string attributeName, const dimensions_t &gridDimensions, 
											Interpolant <Scalar, ArrayType, VectorType> *pScalarInterpolant = NULL);

			/** Attribute addition functions. This function  adds the same attribute in particles data and an accumulator
			* in particles to grid class. These attributes are linked by name and can be accesses together in the
			* advection phase.*/
			bool addIntBasedAttribute(string attributeName) {
				return m_pParticlesToGrid->addIntAttribute(attributeName) &&
						m_pParticlesData->addIntBasedAttribute(attributeName);
			}
			#pragma endregion AccessFunctions

			
		protected:

			#pragma region ClassMembers
			/** Params. TODO: replace all references by params-stored pointers */
			params_t *m_pParams;

			/** Grid Data */
			GridData<VectorType> *m_pGridData;

			/** Velocity field interpolant */
			Interpolant<VectorType, ArrayType, VectorType> *m_pVelocityInterpolant;

			/** Particles data class: will be internally initialized by the particles sampler*/
			ParticlesData<VectorType> *m_pParticlesData;

			/** Particles sampler class: will initialize particles position and, if needed, resample them */
			ParticlesSampler<VectorType, ArrayType> *m_pParticlesSampler;

			/** Particles sampler class: will initialize particles position and, if needed, resample them */
			PositionIntegrator<VectorType, ArrayType> *m_pParticlesIntegrator;

			/** Transfers the information from the grid to the particles, usually it is based on interpolation */
			GridToParticles<VectorType, ArrayType> *m_pGridToParticles;

			/** Transfers the information from the particles to the grid, usually it is based on radial kernels */
			ParticlesToGrid<VectorType, ArrayType> *m_pParticlesToGrid;

			#pragma endregion ClassMembers

			#pragma region InitializationFunctions
			ParticlesSampler<VectorType, ArrayType> * initializeParticlesSampler();
			PositionIntegrator<VectorType, ArrayType> * initializeParticlesIntegrator();
			GridToParticles<VectorType, ArrayType> * initializeGridToParticles();
			ParticlesToGrid<VectorType, ArrayType> * initializeParticlesToGrid();
			#pragma endregion 
		};
	}
}

#endif
