#include "ParticleBased/ParticlesToGrid.h"

namespace Chimera {
	namespace Advection {
		
		#pragma region PrivateFunctionalities
		template<class VectorType, template <class> class ArrayType>
		void ParticlesToGrid<VectorType, ArrayType>::flushAccumulatedVelocities() {	
			accumulatedEntry<VectorType> zeroEntry; //Default vector constructor already sets the vector components to zero
			//zeroEntry.weight = 0.0f;
			m_accVelocityField.assign(zeroEntry);
		}

		template<class VectorType, template <class> class ArrayType>
		void ParticlesToGrid<VectorType, ArrayType>::flushAccumulatedVectorAttributes() {
			accumulatedEntry<VectorType> zeroEntry; //Default vector constructor already sets the vector components to zero
			//zeroEntry.weight = 0.0f;

			for (auto it = m_accVectorFields.begin(); it != m_accVectorFields.end(); ++it) {
				it->second.assign(zeroEntry);
			}
		}

		template<class VectorType, template <class> class ArrayType>
		void ParticlesToGrid<VectorType, ArrayType>::flushAccumulatedScalarAttributes() {
			accumulatedEntry<Scalar> zeroScalarEntry;
			zeroScalarEntry.entry = zeroScalarEntry.weight = 0.0f;

			for (auto it = m_accScalarFields.begin(); it != m_accScalarFields.end(); ++it) {
				it->second.assign(zeroScalarEntry);
			}
		}

		template<class VectorType, template <class> class ArrayType>
		void ParticlesToGrid<VectorType, ArrayType>::flushAccumulatedIntegerAttributes() {
			accumulatedEntry<int> zeroIntegerEntry;
			zeroIntegerEntry.entry = 0;
			zeroIntegerEntry.weight = 0.0f;

			for (auto it = m_accIntegerFields.begin(); it != m_accIntegerFields.end(); ++it) {
				it->second.assign(zeroIntegerEntry);
			}
		}

		#pragma endregion

		template<class VectorType, template <class> class ArrayType>
		void ParticlesToGrid<VectorType, ArrayType>::accumulateVelocity(int ithParticle, const VectorType &weight, 
																		const vector<VectorType> &particleVelocities,
																		const dimensions_t & gridNodeIndex, 
																		velocityComponent_t velocityComponent /* = -1 */) {
			if (velocityComponent != fullVector) {
				m_accVelocityField(gridNodeIndex).entry[velocityComponent] += particleVelocities[ithParticle][velocityComponent] * weight[velocityComponent];
				m_accVelocityField(gridNodeIndex).weight[velocityComponent] += weight[velocityComponent];
			}
			else {
				m_accVelocityField(gridNodeIndex).entry += particleVelocities[ithParticle] * weight;
				m_accVelocityField(gridNodeIndex).weight += weight;
			}	
		}

		template<class VectorType, template <class> class ArrayType>
		void ParticlesToGrid<VectorType, ArrayType>::accumulateScalarField(int ithParticle, Scalar weight, ScalarArray &scalarFieldAccumulator,
																			const vector<Scalar> &particlesScalarAttribute,
																			const dimensions_t & gridNodeIndex) {
			
			Scalar accumulatedValue = particlesScalarAttribute[ithParticle] * weight;
			scalarFieldAccumulator(gridNodeIndex).entry += particlesScalarAttribute[ithParticle] * weight;
			scalarFieldAccumulator(gridNodeIndex).weight += weight;
		}

		template ParticlesToGrid<Vector2, Array2D>;
		template ParticlesToGrid<Vector3, Array3D>;
	}
}