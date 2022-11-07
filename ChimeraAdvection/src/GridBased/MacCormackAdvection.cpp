#include "GridBased/MacCormackAdvection.h"


namespace Chimera {

	namespace Advection {
		
		#pragma region UpdateFunctions
		template <>
		void MacCormackAdvection<Vector2, Array2D>::advect(Scalar dt) {
			GridData2D *pGridData = dynamic_cast<GridData2D*>(m_pGridData);
			Scalar dx = m_pGridData->getGridSpacing();
			
			/** Standard Semi-Lagrangian step first */
			SemiLagrangianAdvection<Vector2, Array2D>::advect(dt);

			/** Do the Semi-Lagrangian step, but this time go forward */
			SemiLagrangianAdvection<Vector2, Array2D>::advect(-dt, &m_auxiliaryVelocityField, m_pVelocityInterpolant->getSibilingInterpolant());

			/** Update limiters */
			for (uint i = 2; i < pGridData->getDimensions().x - 2; i++) {
				for (uint j = 2; j < pGridData->getDimensions().y - 2; j++) {
					m_velocityMinLimiters(i, j).x = getMinLimiterX(i, j);
					m_velocityMinLimiters(i, j).y = getMinLimiterY(i, j);

					m_velocityMaxLimiters(i, j).x = getMaxLimiterX(i, j);
					m_velocityMaxLimiters(i, j).y = getMaxLimiterY(i, j);
				}
			}

			/** Apply corrections */
			#pragma omp parallel for
			for (uint i = 2; i < pGridData->getDimensions().x - 2; i++) {
				for (uint j = 2; j < pGridData->getDimensions().y - 2; j++) {
					Vector2 advectedVelocity;
					Scalar correction = (pGridData->getVelocity(i, j).x - m_auxiliaryVelocityField(i, j).x)*0.5f;
					advectedVelocity.x = pGridData->getAuxiliaryVelocity(i, j).x + correction;

					correction = (pGridData->getVelocity(i, j).y - m_auxiliaryVelocityField(i, j).y)*0.5f;
					advectedVelocity.y = pGridData->getAuxiliaryVelocity(i, j).y + correction;

					//Limit with backward position information, X component
					{
						Vector2 position = Vector2(i, j + 0.5)*dx;
						Vector2 velocity = m_pVelocityInterpolant->interpolate(position);
						Vector2 gridPosition = (m_pPositionIntegrator->integrate(position, velocity, dt)) / dx;

						//** Apply limiters */
						if (advectedVelocity.x > m_velocityMaxLimiters(gridPosition.x, gridPosition.y).x ||
							advectedVelocity.x < m_velocityMinLimiters(gridPosition.x, gridPosition.y).x) {
							advectedVelocity.x = pGridData->getAuxiliaryVelocity(i, j).x;
						}
					}
					
					//Limit with backward position information, Y component
					{
						Vector2 position = Vector2(i + 0.5, j)*dx;
						Vector2 velocity = m_pVelocityInterpolant->interpolate(position);
						Vector2 gridPosition = (m_pPositionIntegrator->integrate(position, velocity, dt)) / dx;

						//** Apply limiters */
						if (advectedVelocity.y > m_velocityMaxLimiters(i, j).y ||
							advectedVelocity.y < m_velocityMinLimiters(i, j).y) {
							advectedVelocity.y = pGridData->getAuxiliaryVelocity(i, j).y;
						}
					}
					

					pGridData->setAuxiliaryVelocity(advectedVelocity, i, j);
				}
			}
		}

		template <>
		void MacCormackAdvection<Vector3, Array3D>::advect(Scalar dt) {
			m_auxiliaryVelocityField.assign(Vector3(0, 0, 0));
			m_velocityMaxLimiters.assign(Vector3(0, 0, 0));
			m_velocityMinLimiters.assign(Vector3(0, 0, 0));

			GridData3D *pGridData = dynamic_cast<GridData3D*>(m_pGridData);
			Scalar dx = m_pGridData->getGridSpacing();
			
			/** Standard Semi-Lagrangian step first */
			SemiLagrangianAdvection<Vector3, Array3D>::advect(dt);

			/** Do the Semi-Lagrangian step, but this time go forward */
			SemiLagrangianAdvection<Vector3, Array3D>::advect(-dt, &m_auxiliaryVelocityField, m_pVelocityInterpolant->getSibilingInterpolant());
			

			/** Update limiters */
			for (uint i = 2; i < pGridData->getDimensions().x - 2; i++) {
				for (uint j = 2; j < pGridData->getDimensions().y - 2; j++) {
					for (uint k = 2; j < pGridData->getDimensions().z - 2; j++) {
						m_velocityMinLimiters(i, j, k).x = getMinLimiterX(i, j, k);
						m_velocityMinLimiters(i, j, k).y = getMinLimiterY(i, j, k);
						m_velocityMinLimiters(i, j, k).z = getMinLimiterZ(i, j, k);

						m_velocityMaxLimiters(i, j, k).x = getMaxLimiterX(i, j, k);
						m_velocityMaxLimiters(i, j, k).y = getMaxLimiterY(i, j, k);
						m_velocityMaxLimiters(i, j, k).z = getMaxLimiterZ(i, j, k);
					}
				}
			}

			/** Apply corrections */
			#pragma omp parallel for
			for (uint i = 2; i < pGridData->getDimensions().x - 2; i++) {
				for (uint j = 2; j < pGridData->getDimensions().y - 2; j++) {
					for (uint k = 2; k < pGridData->getDimensions().z - 2; k++) {
						Vector3 advectedVelocity;
						Scalar correction = (pGridData->getVelocity(i, j, k).x - m_auxiliaryVelocityField(i, j, k).x)*0.5f;
						advectedVelocity.x = pGridData->getAuxiliaryVelocity(i, j, k).x + correction;

						correction = (pGridData->getVelocity(i, j, k).y - m_auxiliaryVelocityField(i, j, k).y)*0.5f;
						advectedVelocity.y = pGridData->getAuxiliaryVelocity(i, j, k).y + correction;

						correction = (pGridData->getVelocity(i, j, k).z - m_auxiliaryVelocityField(i, j, k).z)*0.5f;
						advectedVelocity.z = pGridData->getAuxiliaryVelocity(i, j, k).z + correction;

						//** Apply limiters */
						if (advectedVelocity.x > m_velocityMaxLimiters(i, j, k).x ||
							advectedVelocity.x < m_velocityMinLimiters(i, j, k).x) {
							advectedVelocity.x = pGridData->getAuxiliaryVelocity(i, j, k).x;
						}

						if (advectedVelocity.y > m_velocityMaxLimiters(i, j, k).y ||
							advectedVelocity.y < m_velocityMinLimiters(i, j, k).y) {
							advectedVelocity.y = pGridData->getAuxiliaryVelocity(i, j, k).y;
						}
						
						if (advectedVelocity.z > m_velocityMaxLimiters(i, j, k).z ||
							advectedVelocity.z < m_velocityMinLimiters(i, j, k).z) {
							advectedVelocity.z = pGridData->getAuxiliaryVelocity(i, j, k).z;
						}

						pGridData->setAuxiliaryVelocity(advectedVelocity, i, j, k);
					}
				}
			}
		}


		template <>
		void MacCormackAdvection<Vector2, Array2D>::postProjectionUpdate(Scalar dt) {
			GridData2D *pGridData = dynamic_cast<GridData2D*>(m_pGridData);

			m_auxiliaryScalarField.assign(0.0f);
			m_scalarFieldMaxLimiters.assign(0.0f);
			m_scalarFieldMinLimiters.assign(0.0f);

			/** Standard Semi-Lagrangian step first */
			advectScalarField(dt, *pGridData->getDensityBuffer().getBufferArray2(), m_pDensityInterpolant);

			/** Do the Semi-Lagrangian step, but this time go forward, store on the second buffer, hows that */
			advectScalarField(-dt, m_auxiliaryScalarField, m_pDensityInterpolant->getSibilingInterpolant());

			/** Update limiters */
			for (uint i = 2; i < pGridData->getDimensions().x - 2; i++) {
				for (uint j = 2; j < pGridData->getDimensions().y - 2; j++) {
					m_scalarFieldMinLimiters(i, j) = getMinLimiter(i, j, *pGridData->getDensityBuffer().getBufferArray1());
					m_scalarFieldMaxLimiters(i, j) = getMaxLimiter(i, j, *pGridData->getDensityBuffer().getBufferArray1());
				}
			}

			/** Apply corrections */
			#pragma omp parallel for
			for (uint i = 2; i < pGridData->getDimensions().x - 2; i++) {
				for (uint j = 2; j < pGridData->getDimensions().y - 2; j++) {
					//Buffer 1 has the previous densities
					Scalar correction = ((*pGridData->getDensityBuffer().getBufferArray1())(i, j) - m_auxiliaryScalarField(i, j))*0.5f;
					Scalar advectedValue = (*pGridData->getDensityBuffer().getBufferArray2())(i, j) + correction;

					//** Apply limiters */
					if (advectedValue > m_scalarFieldMaxLimiters(i, j) || advectedValue < m_scalarFieldMinLimiters(i, j)) {
						advectedValue = (*pGridData->getDensityBuffer().getBufferArray2())(i, j);
					}

					pGridData->getDensityBuffer().setValue(advectedValue, i, j);
				}
			}


			/** If temperature interpolant is set */
			if (m_pTemperatureInterpolant) {
				m_auxiliaryScalarField.assign(0.0f);
				m_scalarFieldMaxLimiters.assign(0.0f);
				m_scalarFieldMinLimiters.assign(0.0f);

				/** Standard Semi-Lagrangian step first */
				advectScalarField(dt, *pGridData->getTemperatureBuffer().getBufferArray2(), m_pTemperatureInterpolant);

				/** Do the Semi-Lagrangian step, but this time go forward, store on the second buffer, hows that */
				advectScalarField(-dt, m_auxiliaryScalarField, m_pTemperatureInterpolant->getSibilingInterpolant());

				/** Update limiters */
				for (uint i = 2; i < pGridData->getDimensions().x - 2; i++) {
					for (uint j = 2; j < pGridData->getDimensions().y - 2; j++) {
						m_scalarFieldMinLimiters(i, j) = getMinLimiter(i, j, *pGridData->getTemperatureBuffer().getBufferArray1());
						m_scalarFieldMaxLimiters(i, j) = getMaxLimiter(i, j, *pGridData->getTemperatureBuffer().getBufferArray1());
					}
				}

				/** Apply corrections */
				#pragma omp parallel for
				for (uint i = 2; i < pGridData->getDimensions().x - 2; i++) {
					for (uint j = 2; j < pGridData->getDimensions().y - 2; j++) {
						//Buffer 1 has the previous densities
						Scalar correction = ((*pGridData->getTemperatureBuffer().getBufferArray1())(i, j) - m_auxiliaryScalarField(i, j))*0.5f;
						Scalar advectedValue = (*pGridData->getTemperatureBuffer().getBufferArray2())(i, j) + correction;

						//** Apply limiters */
						if (advectedValue > m_scalarFieldMaxLimiters(i, j) || advectedValue < m_scalarFieldMinLimiters(i, j)) {
							advectedValue = (*pGridData->getTemperatureBuffer().getBufferArray2())(i, j);
						}

						pGridData->getTemperatureBuffer().setValue(advectedValue, i, j);
					}
				}
			}
		}

		template <>
		void MacCormackAdvection<Vector3, Array3D>::postProjectionUpdate(Scalar dt) {
			GridData3D *pGridData = dynamic_cast<GridData3D*>(m_pGridData);

			m_auxiliaryScalarField.assign(0.0f);
			m_scalarFieldMaxLimiters.assign(0.0f);
			m_scalarFieldMinLimiters.assign(0.0f);

			/** Standard Semi-Lagrangian step first */
			advectScalarField(dt, *pGridData->getDensityBuffer().getBufferArray2(), m_pDensityInterpolant);

			/** Do the Semi-Lagrangian step, but this time go forward, store on the second buffer, hows that */
			advectScalarField(-dt, m_auxiliaryScalarField, m_pDensityInterpolant);

			/** Update limiters */
			for (uint i = 2; i < pGridData->getDimensions().x - 2; i++) {
				for (uint j = 2; j < pGridData->getDimensions().y - 2; j++) {
					for (uint k = 2; j < pGridData->getDimensions().z - 2; j++) {
						m_scalarFieldMinLimiters(i, j, k) = getMinLimiter(i, j, k, *pGridData->getDensityBuffer().getBufferArray1());
						m_scalarFieldMaxLimiters(i, j, k) = getMaxLimiter(i, j, k, *pGridData->getDensityBuffer().getBufferArray1());
					}
				}
			}

			/** Apply corrections */
			#pragma omp parallel for
			for (uint i = 2; i < pGridData->getDimensions().x - 2; i++) {
				for (uint j = 2; j < pGridData->getDimensions().y - 2; j++) {
					for (uint k = 2; k < pGridData->getDimensions().z - 2; k++) {
						//Buffer 1 has the previous densities
						Scalar correction = ((*pGridData->getDensityBuffer().getBufferArray1())(i, j, k) - m_auxiliaryScalarField(i, j, k))*0.5f;
						Scalar advectedValue = (*pGridData->getDensityBuffer().getBufferArray2())(i, j, k) + correction;
						
						//** Apply limiters */
						if (advectedValue > m_scalarFieldMaxLimiters(i, j, k) || advectedValue < m_scalarFieldMinLimiters(i, j, k)) {
							advectedValue = (*pGridData->getDensityBuffer().getBufferArray2())(i, j, k);
						}

						pGridData->getDensityBuffer().setValue(advectedValue, i, j, k);
					}
				}
			}


			/** If temperature interpolant is set */
			if (m_pTemperatureInterpolant) {
				m_auxiliaryScalarField.assign(0.0f);
				m_scalarFieldMaxLimiters.assign(0.0f);
				m_scalarFieldMinLimiters.assign(0.0f);

				/** Standard Semi-Lagrangian step first */
				advectScalarField(dt, *pGridData->getTemperatureBuffer().getBufferArray2(), m_pTemperatureInterpolant);

				/** Do the Semi-Lagrangian step, but this time go forward, store on the second buffer, hows that */
				advectScalarField(-dt, m_auxiliaryScalarField, m_pTemperatureInterpolant);

				/** Update limiters */
				for (uint i = 2; i < pGridData->getDimensions().x - 2; i++) {
					for (uint j = 2; j < pGridData->getDimensions().y - 2; j++) {
						for (uint k = 2; j < pGridData->getDimensions().z - 2; j++) {
							m_scalarFieldMinLimiters(i, j, k) = getMinLimiter(i, j, k, *pGridData->getTemperatureBuffer().getBufferArray1());
							m_scalarFieldMaxLimiters(i, j, k) = getMaxLimiter(i, j, k, *pGridData->getTemperatureBuffer().getBufferArray1());
						}
					}
				}

				/** Apply corrections */
				#pragma omp parallel for
				for (uint i = 2; i < pGridData->getDimensions().x - 2; i++) {
					for (uint j = 2; j < pGridData->getDimensions().y - 2; j++) {
						for (uint k = 2; k < pGridData->getDimensions().z - 2; k++) {
							//Buffer 1 has the previous densities
							Scalar correction = ((*pGridData->getTemperatureBuffer().getBufferArray1())(i, j, k) - m_auxiliaryScalarField(i, j, k))*0.5f;
							Scalar advectedValue = (*pGridData->getTemperatureBuffer().getBufferArray2())(i, j, k) + correction;
						
							//** Apply limiters */
							if (advectedValue > m_scalarFieldMaxLimiters(i, j, k) || advectedValue < m_scalarFieldMinLimiters(i, j, k)) {
								advectedValue = (*pGridData->getTemperatureBuffer().getBufferArray2())(i, j, k);
							}

							pGridData->getTemperatureBuffer().setValue(advectedValue, i, j, k);
						}
					}
				}
			}

		}
		#pragma endregion
		
		#pragma region PrivateFunctionalities
		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMinLimiterX(uint i, uint j) {
			GridData2D *pGridData = dynamic_cast<GridData2D*>(m_pGridData);
			Scalar limiter = std::min(pGridData->getVelocity(i, j).x, pGridData->getVelocity(i + 1, j).x);
			limiter = std::min(limiter, pGridData->getVelocity(i, j + 1).x);
			limiter = std::min(limiter, pGridData->getVelocity(i + 1, j + 1).x);

			return limiter;
		}
		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMaxLimiterX(uint i, uint j) {
			GridData2D *pGridData = dynamic_cast<GridData2D*>(m_pGridData);
			Scalar limiter = std::max(pGridData->getVelocity(i, j).x, pGridData->getVelocity(i + 1, j).x);
			limiter = std::max(limiter, pGridData->getVelocity(i, j + 1).x);
			limiter = std::max(limiter, pGridData->getVelocity(i + 1, j + 1).x);

			return limiter;
		}
		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMinLimiterY(uint i, uint j) {
			GridData2D *pGridData = dynamic_cast<GridData2D*>(m_pGridData);
			Scalar limiter = std::min(pGridData->getVelocity(i, j).y, pGridData->getVelocity(i + 1, j).y);
			limiter = std::min(limiter, pGridData->getVelocity(i, j + 1).y);
			limiter = std::min(limiter, pGridData->getVelocity(i + 1, j + 1).y);

			return limiter;
		}
		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMaxLimiterY(uint i, uint j) {
			GridData2D *pGridData = dynamic_cast<GridData2D*>(m_pGridData);
			Scalar limiter = std::max(pGridData->getVelocity(i, j).y, pGridData->getVelocity(i + 1, j).y);
			limiter = std::max(limiter, pGridData->getVelocity(i, j + 1).y);
			limiter = std::max(limiter, pGridData->getVelocity(i + 1, j + 1).y);

			return limiter;
		}

		/** 3-D Limiters */
		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMinLimiterX(uint i, uint j, uint k) {
			GridData3D *pGridData = dynamic_cast<GridData3D*>(m_pGridData);

			Scalar limiter = std::min(pGridData->getVelocity(i, j, k).x, pGridData->getVelocity(i + 1, j, k).x);
			limiter = std::min(limiter, pGridData->getVelocity(i, j + 1, k).x);
			limiter = std::min(limiter, pGridData->getVelocity(i + 1, j + 1, k).x);

			limiter = std::min(limiter, pGridData->getVelocity(i, j, k + 1).x);
			limiter = std::min(limiter, pGridData->getVelocity(i + 1, j, k + 1).x);
			limiter = std::min(limiter, pGridData->getVelocity(i, j + 1, k + 1).x);
			limiter = std::min(limiter, pGridData->getVelocity(i + 1, j + 1, k + 1).x);

			return limiter;
		}

		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMaxLimiterX(uint i, uint j, uint k) {
			GridData3D *pGridData = dynamic_cast<GridData3D*>(m_pGridData);

			Scalar limiter = std::max(pGridData->getVelocity(i, j, k).x, pGridData->getVelocity(i + 1, j, k).x);
			limiter = std::max(limiter, pGridData->getVelocity(i, j + 1, k).x);
			limiter = std::max(limiter, pGridData->getVelocity(i + 1, j + 1, k).x);

			limiter = std::max(limiter, pGridData->getVelocity(i, j, k + 1).x);
			limiter = std::max(limiter, pGridData->getVelocity(i + 1, j, k + 1).x);
			limiter = std::max(limiter, pGridData->getVelocity(i, j + 1, k + 1).x);
			limiter = std::max(limiter, pGridData->getVelocity(i + 1, j + 1, k + 1).x);

			return limiter;
		}

		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMinLimiterY(uint i, uint j, uint k) {
			GridData3D *pGridData = dynamic_cast<GridData3D*>(m_pGridData);

			Scalar limiter = std::min(pGridData->getVelocity(i, j, k).y, pGridData->getVelocity(i + 1, j, k).y);
			limiter = std::min(limiter, pGridData->getVelocity(i, j + 1, k).y);
			limiter = std::min(limiter, pGridData->getVelocity(i + 1, j + 1, k).y);

			limiter = std::min(limiter, pGridData->getVelocity(i, j, k + 1).y);
			limiter = std::min(limiter, pGridData->getVelocity(i + 1, j, k + 1).y);
			limiter = std::min(limiter, pGridData->getVelocity(i, j + 1, k + 1).y);
			limiter = std::min(limiter, pGridData->getVelocity(i + 1, j + 1, k + 1).y);

			return limiter;
		}

		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMaxLimiterY(uint i, uint j, uint k) {
			GridData3D *pGridData = dynamic_cast<GridData3D*>(m_pGridData);

			Scalar limiter = std::max(pGridData->getVelocity(i, j, k).y, pGridData->getVelocity(i + 1, j, k).y);
			limiter = std::max(limiter, pGridData->getVelocity(i, j + 1, k).y);
			limiter = std::max(limiter, pGridData->getVelocity(i + 1, j + 1, k).y);

			limiter = std::max(limiter, pGridData->getVelocity(i, j, k + 1).y);
			limiter = std::max(limiter, pGridData->getVelocity(i + 1, j, k + 1).y);
			limiter = std::max(limiter, pGridData->getVelocity(i, j + 1, k + 1).y);
			limiter = std::max(limiter, pGridData->getVelocity(i + 1, j + 1, k + 1).y);

			return limiter;
		}

		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMinLimiterZ(uint i, uint j, uint k) {
			GridData3D *pGridData = dynamic_cast<GridData3D*>(m_pGridData);
			Scalar limiter = std::min(pGridData->getVelocity(i, j, k)[2], pGridData->getVelocity(i + 1, j, k)[2]);
			limiter = std::min(limiter, pGridData->getVelocity(i, j + 1, k)[2]);
			limiter = std::min(limiter, pGridData->getVelocity(i + 1, j + 1, k)[2]);

			limiter = std::min(limiter, pGridData->getVelocity(i, j, k + 1)[2]);
			limiter = std::min(limiter, pGridData->getVelocity(i + 1, j, k + 1)[2]);
			limiter = std::min(limiter, pGridData->getVelocity(i, j + 1, k + 1)[2]);
			limiter = std::min(limiter, pGridData->getVelocity(i + 1, j + 1, k + 1)[2]);

			return limiter;
		}

		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMaxLimiterZ(uint i, uint j, uint k) {
			GridData3D *pGridData = dynamic_cast<GridData3D*>(m_pGridData);

			Scalar limiter = std::max(pGridData->getVelocity(i, j, k)[2], pGridData->getVelocity(i + 1, j, k)[2]);
			limiter = std::max(limiter, pGridData->getVelocity(i, j + 1, k)[2]);
			limiter = std::max(limiter, pGridData->getVelocity(i + 1, j + 1, k)[2]);

			limiter = std::max(limiter, pGridData->getVelocity(i, j, k + 1)[2]);
			limiter = std::max(limiter, pGridData->getVelocity(i + 1, j, k + 1)[2]);
			limiter = std::max(limiter, pGridData->getVelocity(i, j + 1, k + 1)[2]);
			limiter = std::max(limiter, pGridData->getVelocity(i + 1, j + 1, k + 1)[2]);

			return limiter;
		}


		/**Scalar field limiters */
		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMinLimiter(uint i, uint j, uint k, const Array3D<Scalar> &scalarField) {
			Scalar limiter = std::min(scalarField(i, j, k), scalarField(i + 1, j, k));
			limiter = std::min(limiter, scalarField(i, j + 1, k));
			limiter = std::min(limiter, scalarField(i + 1, j + 1, k));
			limiter = std::min(limiter, scalarField(i, j, k + 1));
			limiter = std::min(limiter, scalarField(i + 1, j, k + 1));
			limiter = std::min(limiter, scalarField(i, j + 1, k + 1));
			limiter = std::min(limiter, scalarField(i + 1, j + 1, k + 1));

			return limiter;
		}

		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMaxLimiter(uint i, uint j, uint k, const Array3D<Scalar> &scalarField) {
			Scalar limiter = std::min(scalarField(i, j, k), scalarField(i + 1, j, k));
			limiter = std::max(limiter, scalarField(i, j + 1, k));
			limiter = std::max(limiter, scalarField(i + 1, j + 1, k));
			limiter = std::max(limiter, scalarField(i, j, k + 1));
			limiter = std::max(limiter, scalarField(i + 1, j, k + 1));
			limiter = std::max(limiter, scalarField(i, j + 1, k + 1));
			limiter = std::max(limiter, scalarField(i + 1, j + 1, k + 1));

			return limiter;
		}


		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMinLimiter(uint i, uint j, const Array2D<Scalar> &scalarField) {
			Scalar limiter = std::min(scalarField(i, j), scalarField(i + 1, j));
			limiter = std::min(limiter, scalarField(i, j + 1));
			limiter = std::min(limiter, scalarField(i + 1, j + 1));
			return limiter;
		}

		template <class VectorType, template <class> class ArrayType>
		Scalar MacCormackAdvection<VectorType, ArrayType>::getMaxLimiter(uint i, uint j, const Array2D<Scalar> &scalarField) {
			Scalar limiter = std::min(scalarField(i, j), scalarField(i + 1, j));
			limiter = std::max(limiter, scalarField(i, j + 1));
			limiter = std::max(limiter, scalarField(i + 1, j + 1));
			return limiter;
		}

		#pragma endregion

		template class MacCormackAdvection<Vector2, Array2D>;
		template class MacCormackAdvection<Vector3, Array3D>;
	}
}
