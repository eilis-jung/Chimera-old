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

#ifndef __CHIMERA_PARTICLES_TO_GRID_H_
#define __CHIMERA_PARTICLES_TO_GRID_H_
#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"
#include "ChimeraCutCells.h"

#include "Kernels/TransferKernel.h"


namespace Chimera {

	using namespace Core;
	using namespace Grids;
	using namespace Particles;
	using namespace Interpolation;
	using namespace CutCells;


	namespace Advection {

		template<class VectorType, template <class> class ArrayType>
		class ParticlesToGrid {
		public:
			/** Holds the accumulated value and its weight together */
			template <class entryType>
			struct accumulatedEntry {
				entryType entry;
				entryType weight;

				accumulatedEntry() {
					
				}
				accumulatedEntry(entryType gEntry, entryType gWeight) {
					entry = gEntry;
					weight = gWeight;
				}
			};

			typedef ArrayType<accumulatedEntry<VectorType>> VectorArray;
			typedef ArrayType<accumulatedEntry<Scalar>>		ScalarArray;
			typedef ArrayType<accumulatedEntry<int>>		IntArray;

		public:
			//Will only reserve sizes inside vectors, explicit position/velocity initialization is on user-side
			ParticlesToGrid(const dimensions_t &gridDimensions, TransferKernel<VectorType> *pKernel) 
				: m_accVelocityField(gridDimensions), m_gridDimensions(gridDimensions) {
				m_pKernel = pKernel;
			}

			/** Velocity transfer from particles to grid. All subclasses must implement this. */
			virtual void transferVelocityToGrid(GridData<VectorType> *pGridData, ParticlesData<VectorType> *pParticlesData) = 0;

			/** Custom attribute transfers. Implementation by subclasses is optional. */
			virtual void transferVectorAttributesToGrid(GridData<VectorType> *pGridData, ParticlesData<VectorType> *pParticlesData) { };
			virtual void transferScalarAttributesToGrid(GridData<VectorType> *pGridData, ParticlesData<VectorType> *pParticlesData) { };
			virtual void transferIntegerAttributesToGrid(GridData<VectorType> *pGridData, ParticlesData<VectorType> *pParticlesData) { };

			#pragma region AcessFunctions 
			/** Adds a vector-based attribute accumulator.*/
			virtual bool addVectorAttribute(string attributeName) {
				if (m_accVectorFields.find(attributeName) != m_accVectorFields.end())
					return false;

				//The size is calculated with the number of particle positions
				m_accVectorFields.insert(std::map<string, VectorArray>::value_type(attributeName, VectorArray(m_gridDimensions)));

				return true;
			}

			/** Adds a scalar-based attribute accumulator.*/
			virtual bool addScalarAttribute(string attributeName) {
				if (m_accScalarFields.find(attributeName) != m_accScalarFields.end())
					return false;

				//The size is calculated with the number of particle positions
				m_accScalarFields.insert(std::map<string, ScalarArray>::value_type(attributeName, ScalarArray(m_gridDimensions)));

				return true;
			}

			/** Adds a scalar-based attribute accumulator on a custom-resolution grid.*/
			virtual bool addScalarAttribute(string attributeName, const dimensions_t &gridDimensions) {
				if (m_accScalarFields.find(attributeName) != m_accScalarFields.end())
					return false;

				//The size is calculated with the number of particle positions
				m_accScalarFields.insert(std::map<string, ScalarArray>::value_type(attributeName, ScalarArray(gridDimensions)));

				return true;
			}

			/** Adds a int-based attribute accumulator.  */
			virtual bool addIntAttribute(string attributeName) {
				if (m_accIntegerFields.find(attributeName) != m_accIntegerFields.end())
					return false;

				m_accIntegerFields.insert(std::map<string, IntArray>::value_type(attributeName, IntArray(m_gridDimensions)));

				return true;
			}

			/** Getters for attributes*/

			/** Returns scalar-based attribute by name. Returns an empty vector if does not find an entry. */
			VectorArray & getVectorAttributeArray(string attributeName) {
				if (m_accVectorFields.find(attributeName) == m_accVectorFields.end())
					addVectorAttribute(attributeName);

				//Cant use [] operator here because of no default constructor
				return m_accVectorFields.find(attributeName)->second;
			}

			ScalarArray & getScalarAttributeArray(string attributeName) {
				if (m_accScalarFields.find(attributeName) == m_accScalarFields.end())
					addScalarAttribute(attributeName);
				
				//Cant use [] operator here because of no default constructor
				return m_accScalarFields.find(attributeName)->second;
			}

			IntArray & getIntAttributeArray(string attributeName) {
				if (m_accIntegerFields.find(attributeName) == m_accIntegerFields.end())
					addIntAttribute(attributeName);

				//Cant use [] operator here because of no default constructor
				return m_accIntegerFields.find(attributeName)->second;
			}
			#pragma endregion

		protected:
			#pragma region PrivateFunctionalities
			
			/** Empties all accumulated fields*/
			virtual void flushAccumulatedVelocities();

			virtual void flushAccumulatedVectorAttributes();
			virtual void flushAccumulatedScalarAttributes();
			virtual void flushAccumulatedIntegerAttributes();

			/** Accumulates a particle velocity into the accumulated vector buffer with a given weight. If the velocity
			  * component is not specified (fullVector), accumulates all possible velocities. */
			virtual void accumulateVelocity(int ithParticle, const VectorType & weight, const vector<VectorType> &particleVelocities,
											const dimensions_t & gridNodeIndex, velocityComponent_t velocityComponent = fullVector);

			virtual void accumulateScalarField(int ithParticle, Scalar weight, ScalarArray &scalarFieldAccumulator,
												const vector<Scalar> &particlesScalarAttribute, const dimensions_t & gridNodeIndex);

			#pragma endregion

			#pragma region ClassMembers 
			dimensions_t m_gridDimensions;

			/** Accumulated velocity field. This is where the particles splat their information. Weights are accumulated
			 ** as well, in order to normalize the values after all particles made their contribution. */
			VectorArray m_accVelocityField;

			/** Transfer kernel */
			TransferKernel<VectorType> *m_pKernel;

			/** Extra maps that might be used for user-defined vector/scalar/integer fields. The name stored in the map
			  * for the accumulator is linked with the particle custom defined attribute. */
			map<string, VectorArray>	m_accVectorFields;
			map<string, ScalarArray>	m_accScalarFields;
			map<string, IntArray>		m_accIntegerFields;

			#pragma endregion
		};
	}
}

#endif