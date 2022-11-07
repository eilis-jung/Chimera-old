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

#ifndef __CHIMERA_PARTICLES_DATA_H_
#define __CHIMERA_PARTICLES_DATA_H_
#pragma once

#include "ChimeraCore.h"

namespace Chimera {

	using namespace Core;

	namespace Particles {

		template<class VectorType>
		class ParticlesData {
		
			public:
				//Will only reserve sizes inside vectors, explicit position/velocity initialization is on user-side
				ParticlesData(int initialNumParticles) { 
					m_numInitialParticles = initialNumParticles;
					m_positions.reserve(initialNumParticles);
					m_velocities.reserve(initialNumParticles);
				}

			#pragma region Acceess Functions
				/** Simple "getters and setters"*/
				const vector<VectorType> & getPositions() const {
					return m_positions;
				}

				vector<VectorType> & getPositions() {
					return m_positions;
				}

				const vector<VectorType> & getVelocities() const {
					return m_velocities;
				}

				vector<VectorType> & getVelocities() {
					return m_velocities;
				}

				const vector<bool> & getResampledParticles() const {
					return m_resampledParticles;
				}

				vector<bool> & getResampledParticles() {
					return m_resampledParticles;
				}

				/** Adds a particle: initializes the position and adds placeholders for all custom allocated variables */
				void addParticle(const VectorType &position) {
					m_positions.push_back(position);
					m_velocities.push_back(VectorType());
					//This is a particle that is initally sampled
					m_resampledParticles.push_back(true);

					for (auto iter = m_vectorBasedAttributes.begin(); iter != m_vectorBasedAttributes.end(); iter++) {
						iter->second.push_back(VectorType());
					}
					for (auto iter = m_scalarBasedAttributes.begin(); iter != m_scalarBasedAttributes.end(); iter++) {
						iter->second.push_back(0.0f);
					}

					for (auto iter = m_integerBasedAttributes.begin(); iter != m_integerBasedAttributes.end(); iter++) {
						iter->second.push_back(0);
					}
				}

				/** Use this function for resampling, since it resets all interior particle fields */
				void resampleParticle(uint particleIndex, const VectorType &position) {
					m_positions[particleIndex] = position;
					VectorType zeroVelocity;
					//Zero out velocity
					m_velocities[particleIndex] = zeroVelocity;

					for (auto iter = m_vectorBasedAttributes.begin(); iter != m_vectorBasedAttributes.end(); iter++) {
						iter->second[particleIndex] = zeroVelocity;
					}
					for (auto iter = m_scalarBasedAttributes.begin(); iter != m_scalarBasedAttributes.end(); iter++) {
						iter->second[particleIndex] = 0.f;
					}

					for (auto iter = m_integerBasedAttributes.begin(); iter != m_integerBasedAttributes.end(); iter++) {
						iter->second[particleIndex] = 0;
					}

					m_resampledParticles[particleIndex] = true;
				}

				/** Adds a vector-based attribute per-particle. If the attribute already exists, attributes are unmodified
				  * and returns false. Otherwise returns true. */
				bool addVectorBasedAttribute(const string & attributeName) {
					if (m_vectorBasedAttributes.find(attributeName) != m_vectorBasedAttributes.end())
						return false;
					
					//The size is calculated with the number of particle positions
					m_vectorBasedAttributes[attributeName].reserve(m_positions.size());

					return true;
				}

				/** Adds a scalar-based attribute per-particle. If the attribute already exists, attributes are unmodified
				  * and returns false. Otherwise returns true. */
				bool addScalarBasedAttribute(const string & attributeName) {
					if (m_scalarBasedAttributes.find(attributeName) != m_scalarBasedAttributes.end())
						return false;

					//The size is calculated with the number of particle positions
					m_scalarBasedAttributes[attributeName].reserve(m_positions.size());
					
					return true;
				}

				/** Adds a int-based attribute per-particle. If the attribute already exists, attributes are unmodified
				  * and returns false. Otherwise returns true. */
				bool addIntBasedAttribute(const string & attributeName) {
					if (m_integerBasedAttributes.find(attributeName) != m_integerBasedAttributes.end())
						return false;

					//The size is calculated with the number of particle positions
					m_integerBasedAttributes[attributeName].reserve(m_positions.size());

					return true;
				}

				/** Returns vector-based attribute by name. Returns an empty vector if does not find an entry. */
				vector<VectorType> & getVectorBasedAttribute(const string & attributeName) {
					if (m_vectorBasedAttributes.find(attributeName) == m_vectorBasedAttributes.end())
						return m_emptyVectorVec;

					return m_vectorBasedAttributes[attributeName];
				}

				/** Returns scalar-based attribute by name. Returns an empty vector if does not find an entry. */
				vector<Scalar> & getScalarBasedAttribute(const string & attributeName) {
					if (m_scalarBasedAttributes.find(attributeName) == m_scalarBasedAttributes.end())
						return m_emptyScalarVec;

					return m_scalarBasedAttributes[attributeName];
				}

				/** Returns integer-based attribute by name. Returns an empty vector if does not find an entry. */
				vector<int> & getIntegerBasedAttribute(const string & attributeName) {
					if (m_integerBasedAttributes.find(attributeName) == m_integerBasedAttributes.end())
						return m_emptyIntegerVec;

					return m_integerBasedAttributes[attributeName];
				}
			
				const vector<int> & getIntegerBasedAttribute(const string & attributeName) const {
					if (m_integerBasedAttributes.find(attributeName) == m_integerBasedAttributes.end())
						return m_emptyIntegerVec;

					return m_integerBasedAttributes[attributeName];
				}

				bool hasIntegerBasedAttribute(const string & attributeName) {
					if (m_integerBasedAttributes.find(attributeName) == m_integerBasedAttributes.end())
						return false;
					return true;
				}

				bool hasScalarBasedAttribute(const string & attributeName) {
					if (m_scalarBasedAttributes.find(attributeName) == m_scalarBasedAttributes.end())
						return false;
					return true;
				}

				bool hasVectorBasedAttribute(const string & attributeName) {
					if (m_vectorBasedAttributes.find(attributeName) == m_vectorBasedAttributes.end())
						return false;
					return true;
				}

				const map<string, vector<VectorType>> & getVectorAttributesMap() const {
					return m_vectorBasedAttributes;
				}

				const map<string, vector<Scalar>> & getScalarAttributesMap() const {
					return m_scalarBasedAttributes;
				}

				const map<string, vector<int>> & getIntAttributesMap() const {
					return m_integerBasedAttributes;
				}

			#pragma endregion
			protected:

				int m_numInitialParticles;
				/* Standard particles properties*/
				vector<VectorType> m_positions;
				vector<VectorType> m_velocities;
				/** Is important to tag if the particles were resampled, for both rendering & simulation purposes */
				vector<bool> m_resampledParticles;

				/* User-side particles attributes: each attribute has an unique name and a vector with the number os particles
				 * associated with it. ParticlesData should own this data, so the user does not have direct pointer control */
				map<string, vector<VectorType>> m_vectorBasedAttributes;
				map<string, vector<Scalar>>		m_scalarBasedAttributes;
				map<string, vector<int>>		m_integerBasedAttributes;

				/* Empty vectors trick. Since getBasedAttribute returns a vector reference, we need static references 
				 * that will return if the getBasedAttribute function fails */
				vector<VectorType> m_emptyVectorVec;
				vector<Scalar> m_emptyScalarVec;
				vector<int> m_emptyIntegerVec;
		};
	}
}
#endif