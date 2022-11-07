#include "GridBased/SemiLagrangianAdvection.h"


namespace Chimera {

	namespace Advection {
		
		#pragma region UpdateFunctioons
		template <>
		void SemiLagrangianAdvection<Vector2, Array2D>::advect(Scalar dt) {
			GridData2D *pGridData = dynamic_cast<GridData2D *>(m_pGridData);
			advect(dt, pGridData->getAuxVelocityArrayPtr(), m_pVelocityInterpolant);
		}

		template <>
		void SemiLagrangianAdvection<Vector3, Array3D>::advect(Scalar dt) {
			GridData3D *pGridData = dynamic_cast<GridData3D *>(m_pGridData);
			advect(dt, pGridData->getAuxVelocityArrayPtr(), m_pVelocityInterpolant);
		}

		template <>
		void SemiLagrangianAdvection<Vector2, Array2D>::postProjectionUpdate(Scalar dt) {
			GridData2D *pGridData = dynamic_cast<GridData2D *>(m_pGridData);
			advectScalarField(dt, *pGridData->getDensityBuffer().getBufferArray2(), m_pDensityInterpolant);
			if (m_pTemperatureInterpolant) {
				advectScalarField(dt, *pGridData->getTemperatureBuffer().getBufferArray2(), m_pTemperatureInterpolant);
			}
		}

		template <>
		void SemiLagrangianAdvection<Vector3, Array3D>::postProjectionUpdate(Scalar dt) {
			GridData3D *pGridData = dynamic_cast<GridData3D *>(m_pGridData);
			advectScalarField(dt, *pGridData->getDensityBuffer().getBufferArray2(), m_pDensityInterpolant);
			if (m_pTemperatureInterpolant) {
				advectScalarField(dt, *pGridData->getTemperatureBuffer().getBufferArray2(), m_pTemperatureInterpolant);
			}
		}
		#pragma endregion

		#pragma region AdvectionFunctions
		template <>
		void SemiLagrangianAdvection<Vector2, Array2D>::advect(Scalar dt, Array2D<Vector2> *pVelocityField, Interpolant<Vector2, Array2D, Vector2> *pVelocityInterpolant) {
			GridData2D *pGridData = dynamic_cast<GridData2D *>(m_pGridData);
			Scalar dx = m_pGridData->getGridSpacing();
			for (int i = 0; i < m_pGridData->getDimensions().x; i++) {
				for (int j = 0; j < m_pGridData->getDimensions().y; j++) {
					Vector2 advectedVelocity;
					//X position
					{
						Vector2 position = Vector2(i, j + 0.5)*dx;
						Vector2 velocity = pVelocityInterpolant->interpolate(position);
						velocity.x = pGridData->getVelocity(i, j).x;
						/** Use negative time-step to go backwards in time */
						position = m_pPositionIntegrator->integrate(position, velocity, -dt, pVelocityInterpolant);
						advectedVelocity.x = pVelocityInterpolant->interpolate(position).x;
					}

					//Y position
					{
						Vector2 position = Vector2(i + 0.5, j)*dx;
						Vector2 velocity = pVelocityInterpolant->interpolate(position);
						velocity.y = pGridData->getVelocity(i, j).y;
						/** Use negative time-step to go backwards in time */
						position = m_pPositionIntegrator->integrate(position, velocity, -dt, pVelocityInterpolant);
						advectedVelocity.y = pVelocityInterpolant->interpolate(position).y;
					}

					(*pVelocityField)(i, j) = advectedVelocity;
				}
			}
		}

		template <>
		void SemiLagrangianAdvection<Vector3, Array3D>::advect(Scalar dt, Array3D<Vector3> *pVelocityField, Interpolant<Vector3, Array3D, Vector3> *pVelocityInterpolant) {
			GridData3D *pGridData = dynamic_cast<GridData3D*>(m_pGridData);
			Scalar dx = m_pGridData->getGridSpacing();
			#pragma omp parallel for
			for (int i = 0; i < m_pGridData->getDimensions().x; i++) {
				for (int j = 0; j < m_pGridData->getDimensions().y; j++) {
					for (int k = 0; k < m_pGridData->getDimensions().z; k++) {
						Vector3 advectedVelocity;
						//X position
						{
							Vector3 position = Vector3(i, j + 0.5, k + 0.5)*dx;
							Vector3 velocity = pVelocityInterpolant->interpolate(position);
							velocity.x = pGridData->getVelocity(i, j, k).x;
							/** Use negative time-step to go backwards in time */
							position = m_pPositionIntegrator->integrate(position, velocity, -dt);
							advectedVelocity.x = pVelocityInterpolant->interpolate(position).x;
						}

						//Y position
						{
							Vector3 position = Vector3(i + 0.5, j, k + 0.5)*dx;
							Vector3 velocity = pVelocityInterpolant->interpolate(position);
							velocity.y = pGridData->getVelocity(i, j, k).y;
							/** Use negative time-step to go backwards in time */
							position = m_pPositionIntegrator->integrate(position, velocity, -dt);
							advectedVelocity.y = pVelocityInterpolant->interpolate(position).y;
						}

						//Z position
						{
							Vector3 position = Vector3(i + 0.5, j + 0.5, k)*dx;
							Vector3 velocity = pVelocityInterpolant->interpolate(position);
							velocity.z = pGridData->getVelocity(i, j, k).z;
							/** Use negative time-step to go backwards in time */
							position = m_pPositionIntegrator->integrate(position, velocity, -dt);
							advectedVelocity.z = pVelocityInterpolant->interpolate(position).z;
						}

						(*pVelocityField)(i, j, k) = advectedVelocity;
					}
				}
			}
		}

		
		template <>
		void SemiLagrangianAdvection<Vector2, Array2D>::advectScalarField(Scalar dt, Array2D<Scalar> &scalarField, Interpolant<Scalar, Array2D, Vector2> *pScalarInterpolant) {
			GridData2D *pGridData = dynamic_cast<GridData2D*>(m_pGridData);
			Scalar dx = m_pGridData->getGridSpacing();

			//#pragma omp parallel for
			for (int i = 0; i < m_pGridData->getDimensions().x; i++) {
				for (int j = 0; j < m_pGridData->getDimensions().y; j++) {
					Vector2 position = Vector2(i + 0.5, j + 0.5)*dx;
					Vector2 velocity = m_pVelocityInterpolant->interpolate(position);

					/** Use negative time-step to go backwards in time */
					position = m_pPositionIntegrator->integrate(position, velocity, -dt);
					scalarField(i, j) = pScalarInterpolant->interpolate(position);
				}
			}
		}

		template <>
		void SemiLagrangianAdvection<Vector3, Array3D>::advectScalarField(Scalar dt, Array3D<Scalar> &scalarField, Interpolant<Scalar, Array3D, Vector3> *pScalarInterpolant) {
			GridData3D *pGridData = dynamic_cast<GridData3D*>(m_pGridData);
			Scalar dx = m_pGridData->getGridSpacing();

			#pragma omp parallel for
			for (int i = 0; i < m_pGridData->getDimensions().x; i++) {
				for (int j = 0; j < m_pGridData->getDimensions().y; j++) {
					for (int k = 0; k < m_pGridData->getDimensions().z; k++) {
						Vector3 position = Vector3(i + 0.5, j + 0.5, k + 0.5)*dx;
						Vector3 velocity = m_pVelocityInterpolant->interpolate(position);
						
						/** Use negative time-step to go backwards in time */
						position = m_pPositionIntegrator->integrate(position, velocity, -dt);
						scalarField(i, j, k) = pScalarInterpolant->interpolate(position);
					}
				}
			}
		}
		#pragma endregion

		template class SemiLagrangianAdvection<Vector2, Array2D>;
		template class SemiLagrangianAdvection<Vector3, Array3D>;
	}
}
