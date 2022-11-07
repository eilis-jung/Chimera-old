#include "Interpolation/BilinearNodalInterpolant3D.h"


namespace Chimera {
	namespace Interpolation {
		
		#pragma region Constructors
		template<class valueType>
		BilinearNodalInterpolant3D<valueType>::BilinearNodalInterpolant3D(const Array3D<valueType>& values, Scalar gridDx)
																					: Interpolant(values) {
			m_dx = gridDx;
		}
		#pragma endregion

		#pragma region Functionalities
		template<>
		/** Scalar-based nodal interpolation in regular grids: assume that the values on the Array3D are stored in nodal locations.
		  * This means no off-set from current position. */
		Scalar BilinearNodalInterpolant3D<Scalar>::interpolate(const Vector3 &position) {
			Vector3 gridSpacePosition(position / m_dx);

			int i = static_cast <int> (floor(gridSpacePosition.x - 0.5f));
			int j = static_cast <int> (floor(gridSpacePosition.y - 0.5f));
			int k = static_cast <int> (floor(gridSpacePosition.z - 0.5f));
			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);
			k = clamp(k, 0, m_values.getDimensions().z - 1);

			Vector3 x1Position(i + 0.5f, j + 0.5f, k + 0.5f);
			Vector3 x2Position(i + 1.5f, j + 1.5f, k + 1.5f);

			int nextI = clamp(i + 1, 0, m_values.getDimensions().x - 1);
			int nextJ = clamp(j + 1, 0, m_values.getDimensions().y - 1);
			int nextK = clamp(k + 1, 0, m_values.getDimensions().z - 1);

			Scalar v1 = m_values(i, j, k) * (x2Position.x - position.x) * (x2Position.y - position.y) +
						m_values(nextI, j, k) * (position.x - x1Position.x) * (x2Position.y - position.y) +
						m_values(i, nextJ, k) * (x2Position.x - position.x) * (position.y - x1Position.y) +
						m_values(nextI, nextJ, k) * (position.x - x1Position.x) * (position.y - x1Position.y);

			Scalar v2 = m_values(i, j, nextK) * (x2Position.x - position.x) * (x2Position.y - position.y) +
						m_values(nextI, j, nextK) * (position.x - x1Position.x) * (x2Position.y - position.y) +
						m_values(i, nextJ, nextK) * (x2Position.x - position.x) * (position.y - x1Position.y) +
						m_values(nextI, nextJ, nextK) * (position.x - x1Position.x) * (position.y - x1Position.y);

			Scalar alpha = gridSpacePosition.z - x1Position.z;
			return v1*(1 - alpha) + v2*alpha;	
		}
		
		/** Vector-based nodal interpolation in regular grids: assume that the values on the Array3D are stored in nodal locations.
		  * This means no off-set from current position. */
		template<class valueType>
		valueType BilinearNodalInterpolant3D<valueType>::interpolate(const Vector3 &position) {
			Vector3 gridSpacePosition(position / m_dx);
			int i = static_cast <int> (floor(gridSpacePosition.x - 0.5f));
			int j = static_cast <int> (floor(gridSpacePosition.y - 0.5f));
			int k = static_cast <int> (floor(gridSpacePosition.z - 0.5f));
			i = clamp(i, 0, m_values.getDimensions().x - 1);
			j = clamp(j, 0, m_values.getDimensions().y - 1);
			k = clamp(k, 0, m_values.getDimensions().z - 1);

			Vector3 x1Position(i + 0.5f, j + 0.5f, k + 0.5f);
			Vector3 x2Position(i + 1.5f, j + 1.5f, k + 1.5f);

			int nextI = clamp(i + 1, 0, m_values.getDimensions().x - 1);
			int nextJ = clamp(j + 1, 0, m_values.getDimensions().y - 1);
			int nextK = clamp(k + 1, 0, m_values.getDimensions().z - 1);

			valueType v1 = m_values(i, j, k) * (x2Position.x - gridSpacePosition.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(nextI, j, k) * (gridSpacePosition.x - x1Position.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(i, nextJ, k) * (x2Position.x - gridSpacePosition.x) * (gridSpacePosition.y - x1Position.y) +
						m_values(nextI, nextJ, k) * (gridSpacePosition.x - x1Position.x) * (gridSpacePosition.y - x1Position.y);

			valueType v2 = m_values(i, j, nextK) * (x2Position.x - gridSpacePosition.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(nextI, j, nextK) * (gridSpacePosition.x - x1Position.x) * (x2Position.y - gridSpacePosition.y) +
						m_values(i, nextJ, nextK) * (x2Position.x - gridSpacePosition.x) * (gridSpacePosition.y - x1Position.y) +
						m_values(nextI, nextJ, nextK) * (gridSpacePosition.x - x1Position.x) * (gridSpacePosition.y - x1Position.y);

			Scalar alpha = gridSpacePosition.z - x1Position.z;
			return v1*(1 - alpha) + v2*alpha;
			//return valueType(v1.x*(1 - alpha) + v2.x*alpha, v1.y*(1 - alpha) + v2.y*alpha, v1.z*(1 - alpha) + v2.z*alpha);
		}


		template<class valueType>
		void BilinearNodalInterpolant3D<valueType>::staggeredToNodeCentered(const Array3D<Vector3> &sourceStaggered, 
																			Array3D<Vector3> &targetNodal) {
			
		}

		template<class valueType>
		void BilinearNodalInterpolant3D<valueType>::staggeredToNodeCentered(const Array3D<Vector3> &sourceStaggered,
																			Array3D<Vector3> &targetNodal,
																			CutVoxels3D<Vector3> *pCutVoxels, bool useAuxVelocities) {
			
			for (int i = 1; i < sourceStaggered.getDimensions().x - 1; i++) {
				for (int j = 1; j < sourceStaggered.getDimensions().y - 1; j++) {
					for (int k = 1; k < sourceStaggered.getDimensions().z - 1; k++) {
						Vector3 nodalVelocity;
						//X component
						bool adjacentCells[4];
						adjacentCells[0] = adjacentCells[1] = adjacentCells[2] = adjacentCells[3] = false;

						if (pCutVoxels->getNodalVertex(i, j, k) != nullptr) {
							Vertex<Vector3> *pVertex = pCutVoxels->getNodalVertex(i, j, k);
							for (int l = 0; l < pVertex->getConnectedFaces().size(); l++) {
								Face<Vector3> *pFace = pVertex->getConnectedFaces()[l];
								if (pFace->getLocation() == YZFace) {
									if(useAuxVelocities)
										nodalVelocity.x += pFace->getAuxiliaryVelocity().x;
									else
										nodalVelocity.x += pFace->getVelocity().x;

									if (pFace->getGridCellLocation().y == j && pFace->getGridCellLocation().x == k) {
										adjacentCells[0] = true;
									}
									else if (pFace->getGridCellLocation().y == j - 1 && pFace->getGridCellLocation().x == k - 1) {
										adjacentCells[2] = true;
									}
									else if (pFace->getGridCellLocation().y == j - 1) {
										adjacentCells[1] = true;
									} else if (pFace->getGridCellLocation().x == k - 1) {
										adjacentCells[3] = true;
									}
								}
							}

							if (!adjacentCells[0]) {
								nodalVelocity.x += sourceStaggered(i, j, k).x;
							}
							if (!adjacentCells[1]) {
								nodalVelocity.x += sourceStaggered(i, j - 1, k).x;
							}
							if (!adjacentCells[2]) {
								nodalVelocity.x += sourceStaggered(i, j - 1, k - 1).x;
							}
							if (!adjacentCells[3]) {
								nodalVelocity.x += sourceStaggered(i, j, k - 1).x;
							}
							nodalVelocity.x *= 0.25;
						}
						else {
							nodalVelocity.x =	sourceStaggered(i, j, k).x + sourceStaggered(i, j - 1, k).x +
												sourceStaggered(i, j - 1, k - 1).x + sourceStaggered(i, j, k - 1).x;
							nodalVelocity.x *= 0.25;
						}

						//Y component
						if (pCutVoxels->getNodalVertex(i, j, k) != nullptr) {
							Vertex<Vector3> *pVertex = pCutVoxels->getNodalVertex(i, j, k);
							for (int l = 0; l < pVertex->getConnectedFaces().size(); l++) {
								Face<Vector3> *pFace = pVertex->getConnectedFaces()[l];
								if (pFace->getLocation() == XZFace) {
									if (useAuxVelocities)
										nodalVelocity.y += pFace->getAuxiliaryVelocity().y;
									else
										nodalVelocity.y += pFace->getVelocity().y;

									if (pFace->getGridCellLocation().x == i && pFace->getGridCellLocation().y == k) {
										adjacentCells[0] = true;
									}
									else if (pFace->getGridCellLocation().x == i - 1 && pFace->getGridCellLocation().y == k - 1) {
										adjacentCells[2] = true;
									}
									else if (pFace->getGridCellLocation().x == i - 1) {
										adjacentCells[1] = true;
									}
									else if (pFace->getGridCellLocation().y == k - 1) {
										adjacentCells[3] = true;
									}
								}
							}

							if (!adjacentCells[0]) {
								nodalVelocity.y += sourceStaggered(i, j, k).y;
							}
							if (!adjacentCells[1]) {
								nodalVelocity.y += sourceStaggered(i - 1, j, k).y;
							}
							if (!adjacentCells[2]) {
								nodalVelocity.y += sourceStaggered(i - 1, j, k - 1).y;
							}
							if (!adjacentCells[3]) {
								nodalVelocity.y += sourceStaggered(i, j, k - 1).y;
							}
							nodalVelocity.y *= 0.25;
						}
						else {
							nodalVelocity.y =	sourceStaggered(i, j, k).y + sourceStaggered(i - 1, j, k).y +
												sourceStaggered(i - 1, j, k - 1).y + sourceStaggered(i, j, k - 1).y;
							nodalVelocity.y *= 0.25;
						}

						//Z component
						if (pCutVoxels->getNodalVertex(i, j, k) != nullptr) {
							Vertex<Vector3> *pVertex = pCutVoxels->getNodalVertex(i, j, k);
							for (int l = 0; l < pVertex->getConnectedFaces().size(); l++) {
								Face<Vector3> *pFace = pVertex->getConnectedFaces()[l];
								if (pFace->getLocation() == XYFace) {
									if (useAuxVelocities)
										nodalVelocity.z += pFace->getAuxiliaryVelocity().z;
									else
										nodalVelocity.z += pFace->getVelocity().z;

									if (pFace->getGridCellLocation().x == i && pFace->getGridCellLocation().y == j) {
										adjacentCells[0] = true;
									}
									else if (pFace->getGridCellLocation().x == i - 1 && pFace->getGridCellLocation().y == j - 1) {
										adjacentCells[2] = true;
									}
									else if (pFace->getGridCellLocation().x == i - 1) {
										adjacentCells[1] = true;
									}
									else if (pFace->getGridCellLocation().y == j - 1) {
										adjacentCells[3] = true;
									}
								}
							}

							if (!adjacentCells[0]) {
								nodalVelocity.z += sourceStaggered(i, j, k).z;
							}
							if (!adjacentCells[1]) {
								nodalVelocity.z += sourceStaggered(i - 1, j, k).z;
							}
							if (!adjacentCells[2]) {
								nodalVelocity.z += sourceStaggered(i - 1, j - 1, k).z;
							}
							if (!adjacentCells[3]) {
								nodalVelocity.z += sourceStaggered(i, j - 1, k).z;
							}
							nodalVelocity.z *= 0.25;
						}
						else {
							nodalVelocity.z =	sourceStaggered(i, j, k).z + sourceStaggered(i, j - 1, k).z +
												sourceStaggered(i - 1, j - 1, k).z + sourceStaggered(i - 1, j, k).z;
							nodalVelocity.z *= 0.25;
						}

						targetNodal(i, j, k) = nodalVelocity;
					}

					
				}
			}
		}
		#pragma endregion

		/** Template linker trickerino for templated classes in CPP*/
		template class BilinearNodalInterpolant3D<Scalar>;
		template class BilinearNodalInterpolant3D<Vector3>;
		template class BilinearNodalInterpolant3D<Vector3D>;
	}
}