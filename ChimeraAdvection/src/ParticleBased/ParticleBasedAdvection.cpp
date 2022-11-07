#include "ParticleBased/ParticleBasedAdvection.h"
#include "Integration/ForwardEulerIntegrator.h"
#include "Integration/RungeKutta2Integrator.h"

#include "ParticleBased/GridToParticlesFLIP2D.h"
#include "ParticleBased/GridToParticlesAPIC2D.h"
#include "ParticleBased/GridToParticlesRPIC2D.h"
#include "ParticleBased/GridToParticlesFLIP3D.h"

#include "ParticleBased/ParticlesToNodalGrid2D.h"
#include "ParticleBased/ParticlesToNodalGrid3D.h"
#include "ParticleBased/ParticlesToStaggeredGrid2D.h"
#include "ParticleBased/ParticlesToStaggeredGrid3D.h"

#include "Kernels/SPHKernel.h"
#include "Kernels/BilinearKernel.h"
#include "Kernels/InverseDistanceKernel.h"

namespace Chimera {
	namespace Advection {


		#pragma region AccessFunctions	
		template <class VectorType, template <class> class ArrayType> 
		bool ParticleBasedAdvection<VectorType, ArrayType>::addScalarBasedAttribute(string attributeName, 
																					Interpolant <Scalar, ArrayType, VectorType> *pScalarInterpolant /*= NULL*/) {
			m_pGridToParticles->addScalarAttribute(attributeName, pScalarInterpolant);
			if (m_pParticlesToGrid->addScalarAttribute(attributeName) && m_pParticlesData->addScalarBasedAttribute(attributeName)) {
				vector<Scalar> &pointsScalar = m_pParticlesData->getScalarBasedAttribute(attributeName);
				const vector<VectorType> &pointsPosition = getParticlesPosition();
				if (pScalarInterpolant) {
					for (int i = 0; i < pointsPosition.size(); i++) {
						Scalar scalarValue = pScalarInterpolant->interpolate(pointsPosition[i]);
						pointsScalar.push_back(scalarValue);
					}
				}
				else {
					for (int i = 0; i < pointsPosition.size(); i++) {
						pointsScalar.push_back(0.0f);
					}
				}
				return true;
			}
			return false;
		}

		template <class VectorType, template <class> class ArrayType>
		bool ParticleBasedAdvection<VectorType, ArrayType>::addScalarBasedAttribute(string attributeName, const dimensions_t &gridDimensions,
			Interpolant <Scalar, ArrayType, VectorType> *pScalarInterpolant /*= NULL*/) {
			m_pGridToParticles->addScalarAttribute(attributeName, pScalarInterpolant);
			if (m_pParticlesToGrid->addScalarAttribute(attributeName, gridDimensions) && m_pParticlesData->addScalarBasedAttribute(attributeName)) {
				vector<Scalar> &pointsScalar = m_pParticlesData->getScalarBasedAttribute(attributeName);
				const vector<VectorType> &pointsPosition = getParticlesPosition();
				if (pScalarInterpolant) {
					for (int i = 0; i < pointsPosition.size(); i++) {
						Scalar interpValue = pScalarInterpolant->interpolate(pointsPosition[i]);
						pointsScalar.push_back(pScalarInterpolant->interpolate(pointsPosition[i]));
					}
				}
				else {
					for (int i = 0; i < pointsPosition.size(); i++) {
						pointsScalar.push_back(0.0f);
					}
				}
				return true;
			}
			return false;
		}
		#pragma endregion 

		#pragma region UpdateFunctions

		template <class VectorType, template <class> class ArrayType>
		void ParticleBasedAdvection<VectorType, ArrayType>::updatePositions(Scalar dt) {
			m_pParticlesIntegrator->integratePositions(dt);
			m_pParticlesSampler->resampleParticles(m_pParticlesData);
		}

		template <class VectorType, template <class> class ArrayType>
		void ParticleBasedAdvection<VectorType, ArrayType>::updateGridAttributes() {
			m_pParticlesToGrid->transferVelocityToGrid(m_pGridData, m_pParticlesData);
			m_pParticlesToGrid->transferScalarAttributesToGrid(m_pGridData, m_pParticlesData);
		}

		template <class VectorType, template <class> class ArrayType>
		void ParticleBasedAdvection<VectorType, ArrayType>::updateParticleAttributes() {
			m_pGridToParticles->transferVelocityToParticles(m_pGridData, m_pParticlesData);
			m_pGridToParticles->transferScalarAttributesToParticles(m_pGridData, m_pParticlesData);
		}
		#pragma endregion

		#pragma region InitializationFunctions
		template<class VectorT, template <class> class ArrayType>
		ParticlesSampler<VectorT, ArrayType> * ParticleBasedAdvection<VectorT, ArrayType>::initializeParticlesSampler() {
			switch (m_pParams->samplingMethod) {
				case stratifiedSampling:
					return new FastParticlesSampler<VectorT, ArrayType>(m_pGridData, m_pParams->particlesPerCell);
				break;

				case poissonSampling:
					return new PoissonParticleSampler<VectorT, ArrayType>(m_pGridData, m_pParams->particlesPerCell);
				break;
			}

			return nullptr;
		}

		template<>
		PositionIntegrator<Vector2, Array2D> * ParticleBasedAdvection<Vector2, Array2D>::initializeParticlesIntegrator() {
			Interpolant<Vector2, Array2D, Vector2> *pVelocityInterpolant;
			GridData2D *pGridData = dynamic_cast<GridData2D *>(m_pGridData);
			Scalar dx = pGridData->getGridSpacing();

			if (m_pParams->positionIntegrationInterpolation == CubicStreamfunction) {
				pVelocityInterpolant = new CubicStreamfunctionInterpolant2D<Vector2>(pGridData->getVelocityArray(), dx);
			}
			else if (m_pParams->positionIntegrationInterpolation == LinearStreamfunction) {
				pVelocityInterpolant  = new BilinearStreamfunctionInterpolant2D<Vector2>(pGridData->getVelocityArray(), dx);
			}
			else if (m_pParams->positionIntegrationInterpolation == SubdivisStreamfunction) {
				pVelocityInterpolant = new SubdivisionStreamfunctionInterpolant2D<Vector2>(pGridData->getVelocityArray(), dx, dx / 8);
			}
			else {
				pVelocityInterpolant = m_pVelocityInterpolant;
			}

			switch (m_pParams->integrationMethod) {
				case forwardEuler:
					return new ForwardEulerIntegrator<Vector2, Array2D>(m_pParticlesSampler->getParticlesData(), pVelocityInterpolant, dx);
				break;

				case RungeKutta_2:
					return new RungeKutta2Integrator<Vector2, Array2D>(m_pParticlesSampler->getParticlesData(), pVelocityInterpolant, dx);
				break;
			}

			return nullptr;
		}

		template<>
		PositionIntegrator<Vector3, Array3D> * ParticleBasedAdvection<Vector3, Array3D>::initializeParticlesIntegrator() {
			Interpolant<Vector3, Array3D, Vector3> *pVelocityInterpolant;
			GridData3D *pGridData = dynamic_cast<GridData3D *>(m_pGridData);
			Scalar dx = pGridData->getGridSpacing();

			//For now not loading for different interpolants
			pVelocityInterpolant = m_pVelocityInterpolant;

			switch (m_pParams->integrationMethod) {
				case forwardEuler:
					return new ForwardEulerIntegrator<Vector3, Array3D>(m_pParticlesSampler->getParticlesData(), pVelocityInterpolant, dx);
				break;

				case RungeKutta_2:
					return new RungeKutta2Integrator<Vector3, Array3D>(m_pParticlesSampler->getParticlesData(), pVelocityInterpolant, dx);
				break;
			}

			return nullptr;
		}

		template<>
		GridToParticles<Vector2, Array2D> * ParticleBasedAdvection<Vector2, Array2D>::initializeGridToParticles() {
			switch (m_pParams->gridToParticleTransferMethod) {
				case params_t::gridToParticle_t::PIC:
					return new GridToParticlesFLIP2D(m_pVelocityInterpolant, 1.0f);
				break;
				case params_t::gridToParticle_t::FLIP:
					return new GridToParticlesFLIP2D(m_pVelocityInterpolant, m_pParams->mixFLIP);
				break;
				case params_t::gridToParticle_t::APIC:
					return new GridToParticlesAPIC2D(m_pVelocityInterpolant);
				break;

				case params_t::gridToParticle_t::RPIC:
					return new GridToParticlesRPIC2D(m_pVelocityInterpolant);
				break;
			}


			/*if (dynamic_cast<GridToParticlesRPIC2D*>(m_pGridToParticlesTransfer)) {
				ParticlesData<Vector2> *pParticlesData = pParticleBasedAdvection->getParticlesData();
				if (!pParticlesData->addVectorBasedAttribute("angularMomentum")) {
					Logger::get() << "Failed to add angular momentum for RPIC" << endl;
				}
				else {
					vector<Vector2> &particleAngularMomentum = pParticlesData->getVectorBasedAttribute("angularMomentum");
					particleAngularMomentum.assign(particleAngularMomentum.capacity(), Vector2());
				}
			}
			else if (dynamic_cast<GridToParticlesAPIC2D*>(m_pGridToParticlesTransfer)) {
				ParticlesData<Vector2> *pParticlesData = pParticleBasedAdvection->getParticlesData();
				if (!pParticlesData->addVectorBasedAttribute("velocityDerivativeX") || !pParticlesData->addVectorBasedAttribute("velocityDerivativeY")) {
					Logger::get() << "Failed to add velocity derivative for APIC" << endl;
				}
				else {
					vector<Vector2> &particleVelocityDerivativeX = pParticlesData->getVectorBasedAttribute("velocityDerivativeX");
					vector<Vector2> &particleVelocityDerivativeY = pParticlesData->getVectorBasedAttribute("velocityDerivativeY");
					particleVelocityDerivativeX.assign(particleVelocityDerivativeX.capacity(), Vector2());
					particleVelocityDerivativeY.assign(particleVelocityDerivativeY.capacity(), Vector2());
				}
			}*/
			return nullptr;
		}

		template<>
		GridToParticles<Vector3, Array3D> * ParticleBasedAdvection<Vector3, Array3D>::initializeGridToParticles() {
			switch (m_pParams->gridToParticleTransferMethod) {
				case params_t::gridToParticle_t::PIC:
					return new GridToParticlesFLIP3D(m_pVelocityInterpolant, 1.0f);
					break;
				case params_t::gridToParticle_t::FLIP:
					return new GridToParticlesFLIP3D(m_pVelocityInterpolant, m_pParams->mixFLIP);
					break;
				case params_t::gridToParticle_t::APIC:
					return nullptr; //TODO
					break;

				case params_t::gridToParticle_t::RPIC:
					return nullptr; //TODO
				break;
			}
			return nullptr;
		}

		template<>
		ParticlesToGrid<Vector2, Array2D> * ParticleBasedAdvection<Vector2, Array2D>::initializeParticlesToGrid() {
			TransferKernel<Vector2> *pTransferKernel = nullptr;
			switch (m_pParams->kernelType)
			{
				case SPHkernel:
					pTransferKernel = new SPHKernel<Vector2>(m_pGridData, m_pParams->kernelSize);
				break;

				case bilinearKernel:
					pTransferKernel = new BilinearKernel<Vector2>(m_pGridData, m_pParams->kernelSize);
				break;

				case inverseDistance:
					pTransferKernel = new InverseDistanceKernel<Vector2>(m_pGridData, m_pParams->kernelSize);
				break;
			}

			switch (m_pParams->gridArrangement) {
				case staggeredArrangement:
					return new ParticlesToStaggeredGrid2D(m_pGridData->getDimensions(), pTransferKernel);
				break;

				case nodalArrangement:
					return new ParticlesToNodalGrid2D(m_pGridData->getDimensions(), pTransferKernel);
				break;
			}
			return nullptr;
		}

		template<>
		ParticlesToGrid<Vector3, Array3D> * ParticleBasedAdvection<Vector3, Array3D>::initializeParticlesToGrid() {
			TransferKernel<Vector3> *pTransferKernel = nullptr;
			switch (m_pParams->kernelType)
			{
			case SPHkernel:
				pTransferKernel = new SPHKernel<Vector3>(m_pGridData, m_pParams->kernelSize);
				break;

			case bilinearKernel:
				pTransferKernel = new BilinearKernel<Vector3>(m_pGridData, m_pParams->kernelSize);
				break;

			case inverseDistance:
				pTransferKernel = new InverseDistanceKernel<Vector3>(m_pGridData, m_pParams->kernelSize);
				break;
			}

			switch (m_pParams->gridArrangement) {
				case staggeredArrangement:
					return new ParticlesToStaggeredGrid3D(m_pGridData->getDimensions(), pTransferKernel);
				break;

				case nodalArrangement:
					return new ParticlesToNodalGrid3D(m_pGridData->getDimensions(), pTransferKernel);
				break;
			}
			return nullptr;
		}

		#pragma endregion 

		template class ParticleBasedAdvection<Vector2, Array2D>;
		template class ParticleBasedAdvection<Vector3, Array3D>;
	}
}