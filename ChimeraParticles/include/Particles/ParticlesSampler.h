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

#ifndef __CHIMERA_PARTICLES_SAMPLER_H_
#define __CHIMERA_PARTICLES_SAMPLER_H_
#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraInterpolation.h"
#include "ChimeraBoundaryConditions.h"
#include "Particles/ParticlesData.h"
#include "ChimeraCutCells.h"

namespace Chimera {

	using namespace Core;
	using namespace Grids;
	using namespace BoundaryConditions;
	namespace Particles {

		/** Simple stratified particle sampler */
		template <class VectorType, template <class> class ArrayType>
		class ParticlesSampler {
			
		public:
			#pragma region Constructors
			ParticlesSampler(GridData<VectorType> *pGridData, Scalar particlesPerCell, ParticlesData<VectorType> *pParticlesData = nullptr) {
				m_pGridData = pGridData;
				m_particlesPerCell = particlesPerCell;
				m_pCutCells = NULL;
				
				if (!m_pParticlesData)
					m_pParticlesData = createSampledParticles();
				else
					m_pParticlesData = pParticlesData;
			}; 

			#pragma region AccessFunctions
			void setCutCells(CutCells::CutCellsBase<VectorType> *pCutcells) {
				m_pCutCells = pCutcells;
			}

			ParticlesData<VectorType> * getParticlesData() {
				return m_pParticlesData;
			}

			void setBoundaryConditions(const vector<BoundaryCondition<VectorType> *> &boundaryConditions) {
				m_boundaryConditions = boundaryConditions;
			}
			#pragma endregion

			#pragma region Functionalities
			/** Particle resampling*/
			virtual void resampleParticles(ParticlesData<VectorType> *pParticlesData);

			/** Particle data individual velocity update*/
			virtual void interpolateVelocities(Interpolation::Interpolant<VectorType, ArrayType, VectorType> *pInterpolant, ParticlesData<VectorType> *pParticlesData);
			#pragma endregion
		
		protected:
			#pragma region PrivateStructures
			typedef struct cellParticleCount_t {
				dimensions_t cellIndex;
				int count;

				cellParticleCount_t(const dimensions_t &gCellIndex, int gCount) {
					cellIndex = gCellIndex;
					count = gCount;
				}
			} cellParticleCount_t;

			struct CountCompareNode : public std::binary_function<cellParticleCount_t, cellParticleCount_t, bool>
			{
				bool operator()(const cellParticleCount_t lhs, const cellParticleCount_t rhs) const
				{
					return lhs.count > rhs.count;
				}
			};
			#pragma endregion

			#pragma region PrivateFunctionalities
			/** Resamples particles that are out of limits from the boundaries. Treats boundaries as periodic */
			virtual void boundaryResample(int ithParticle, ParticlesData<VectorType> *pParticlesData);

			/** Particle Creation Function */
			virtual ParticlesData<VectorType> * createSampledParticles();
			#pragma endregion

			#pragma region ClassMembers
			GridData<VectorType> *m_pGridData;
			int m_particlesPerCell;
			CutCells::CutCellsBase<VectorType> *m_pCutCells;
			ParticlesData<VectorType> * m_pParticlesData;
			/** Boundary conditions that may help particles resampling */
			vector<BoundaryCondition<VectorType> *> m_boundaryConditions;
			#pragma endregion
		};
	}
}

#endif