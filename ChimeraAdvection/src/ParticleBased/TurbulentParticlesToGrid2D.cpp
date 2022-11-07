#include "ParticleBased/TurbulentParticlesToGrid2D.h"

namespace Chimera {


	namespace Advection {

		#pragma region PrivateFunctionalities
		
		void TurbulentParticlesGrid2D::accumulateScalarFieldValues(string scalarFieldName, ParticlesData<Vector2> *pParticlesData, Scalar dx) {
			
			if (scalarFieldName == "streamfunctionFine") {
				ScalarArray &scalarFieldAccumulator = getScalarAttributeArray(scalarFieldName);
				int numSubdivisions = scalarFieldAccumulator.getDimensions().x / m_gridDimensions.x;
				const vector<Scalar> &particlesScalarAttribute = pParticlesData->getScalarBasedAttribute(scalarFieldName);

				Scalar fineGridDx = dx / numSubdivisions;
				for (int ithParticle = 0; ithParticle < pParticlesData->getPositions().size(); ithParticle++) {

					Scalar currValue = particlesScalarAttribute[ithParticle];
					Vector2 transformedPosition = (pParticlesData->getPositions()[ithParticle] / dx)* numSubdivisions;
					int i, j;
					Vector2 relativeFractions;

					i = floor(transformedPosition.x);
					j = floor(transformedPosition.y);

					relativeFractions.x = transformedPosition.x - i;
					relativeFractions.y = transformedPosition.y - j;

					Scalar totalWeight = 0;
					Scalar weight = (1 - relativeFractions.x)*(1 - relativeFractions.y);
					totalWeight += weight;
					Vector2 destPosition = Vector2(i, j);
					//weight = m_pKernel->calculateKernel(transformedPosition, destPosition, (transformedPosition - destPosition).length()*fineGridDx);
					accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i, j));

					weight = relativeFractions.x*(1 - relativeFractions.y);
					totalWeight += weight;
					destPosition = Vector2(i + 1, j);
					//weight = m_pKernel->calculateKernel(transformedPosition, destPosition, (transformedPosition - destPosition).length()*fineGridDx);
					if (i < m_gridDimensions.x - 1) {
						accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i + 1, j));
					}

					weight = (1 - relativeFractions.x)*relativeFractions.y;
					totalWeight += weight;
					destPosition = Vector2(i, j + 1);
					//weight = m_pKernel->calculateKernel(transformedPosition, destPosition, (transformedPosition - destPosition).length()*fineGridDx);
					if (j < m_gridDimensions.y - 1) {
						accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i, j + 1));
					}

					weight = relativeFractions.x*relativeFractions.y;
					totalWeight += weight;
					destPosition = Vector2(i + 1, j + 1);
					//weight = m_pKernel->calculateKernel(transformedPosition, destPosition, (transformedPosition - destPosition).length()*fineGridDx);
					if (i < m_gridDimensions.x - 1 && j < m_gridDimensions.y - 1) {
						accumulateScalarField(ithParticle, weight, scalarFieldAccumulator, particlesScalarAttribute, dimensions_t(i + 1, j + 1));
					}
				}
			}
			else {
				ParticlesToStaggeredGrid2D::accumulateScalarFieldValues(scalarFieldName, pParticlesData, dx);
			}
		}
		#pragma endregion
	}
}
