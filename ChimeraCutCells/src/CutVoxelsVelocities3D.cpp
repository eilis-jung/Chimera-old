#include "CutCells/CutVoxelsVelocities3D.h"
#include "ChimeraEigenWrapper.h"

namespace Chimera {
	namespace CutCells {

		#pragma region Constructors
		CutVoxelsVelocities3D::CutVoxelsVelocities3D(CutVoxels3D<Vector3> *pCutVoxels, solidBoundaryType_t solidBoundaryType)
			: CutCellsVelocities(pCutVoxels, solidBoundaryType) {
			m_pCutVoxels = pCutVoxels;
			m_mixedNodeInterpolationType = Unweighted;
		}
		#pragma endregion 

		#pragma region Functionalities

		void CutVoxelsVelocities3D::update(const Array3D<Vector3> &nodalVelocities, bool useAuxiliaryVelocities) {
			if (m_solidBoundary == Solid_NoSlip) {
				processNoSlipVelocities(nodalVelocities, useAuxiliaryVelocities);
			}
			else if (m_solidBoundary == Solid_FreeSlip) {
				processNoSlipVelocities(nodalVelocities, useAuxiliaryVelocities);
				processFreeSlipVelocities(nodalVelocities, useAuxiliaryVelocities);
			}
		}

		#pragma endregion 

		#pragma region PrivateFunctionalities
		void CutVoxelsVelocities3D::processNoSlipVelocities(const Array3D<Vector3> &nodalVelocities, bool useAuxiliaryVelocities) {
			/** TODO: process vertices more efficiently */
			if (m_pCutVoxels) {
				Scalar dx = m_pCutVoxels->getGridSpacing();
				for (int i = 0; i < m_pCutVoxels->getNumberCutVoxels(); i++) {
					auto cutVoxel = m_pCutVoxels->getCutVoxel(i);
					const map<uint, Vertex<Vector3> *> & verticesMap = cutVoxel.getVerticesMap();
					for (auto it = verticesMap.begin(); it != verticesMap.end(); it++) {
						if (it->second->getVertexType() == gridVertex) {
							Vector3 tempPoint = it->second->getPosition() / dx;
							if (useAuxiliaryVelocities)
								it->second->setAuxiliaryVelocity(nodalVelocities(floor(tempPoint.x), floor(tempPoint.y), floor(tempPoint.z)));
							else
								it->second->setVelocity(nodalVelocities(floor(tempPoint.x), floor(tempPoint.y), floor(tempPoint.z)));
						}
					}
				}
			}
			
		}
		void CutVoxelsVelocities3D::processFreeSlipVelocities(const Array3D<Vector3> &nodalVelocities, bool useAuxiliaryVelocities) {
			/** First process mixed nodes velocities */
			for (int i = 0; i < m_pCutVoxels->getNumberCutVoxels(); i++) {
				auto cutVoxel = m_pCutVoxels->getCutVoxel(i);
				const map<uint, Vertex<Vector3> *> & verticesMap = cutVoxel.getVerticesMap();
				//Reset vertices update
				for (auto it = verticesMap.begin(); it != verticesMap.end(); it++) {
					it->second->setUpdated(false);
				}
				for (auto it = verticesMap.begin(); it != verticesMap.end(); it++) {
					if (it->second->getVertexType() == edgeVertex && !it->second->hasUpdated()) {
						if (useAuxiliaryVelocities) {
							it->second->setAuxiliaryVelocity(interpolateMixedNodeVelocitiesUnweighted(it->second, useAuxiliaryVelocities));
						}
						else {
							it->second->setVelocity(interpolateMixedNodeVelocitiesUnweighted(it->second, useAuxiliaryVelocities));
						}
						it->second->setUpdated(true);
					}
				}
			}

			/** Then spread mixed node velocities to face velocities */

		}

		Vector3 CutVoxelsVelocities3D::interpolateMixedNodeVelocitiesUnweighted(Vertex<Vector3>* pVertex, bool useAuxiliaryVelocities) {

			MatrixNxND leastSquaresMat(pVertex->getConnectedHalfFaces().size(), 3);
			MatrixNxND fluxMat(pVertex->getConnectedHalfFaces().size(), 1);
			for (int i = 0; i < pVertex->getConnectedHalfFaces().size(); i++) {
				Vector3 normals = pVertex->getConnectedHalfFaces()[i]->getNormal();
				for (int j = 0; j < 3; j++) {
				
					leastSquaresMat(i, j) = pVertex->getConnectedHalfFaces()[i]->getNormal()[j];
				}
				if(useAuxiliaryVelocities)
					fluxMat(i, 0) = pVertex->getConnectedHalfFaces()[i]->getNormal().dot(pVertex->getConnectedHalfFaces()[i]->getFace()->getAuxiliaryVelocity());
				else
					fluxMat(i, 0) = pVertex->getConnectedHalfFaces()[i]->getNormal().dot(pVertex->getConnectedHalfFaces()[i]->getFace()->getVelocity());
			}

			MatrixNxND leastSquaresMatTranspose(leastSquaresMat);
			leastSquaresMatTranspose.transpose();

			MatrixNxND finalFluxMat(leastSquaresMatTranspose*fluxMat); //3x1 matrix
			Matrix3x3D normalMatrix(leastSquaresMatTranspose*leastSquaresMat);
			normalMatrix.invert();

			Vector3 mixedNodeVel = convertToVector3F(normalMatrix*Vector3D(finalFluxMat(0, 0), finalFluxMat(1, 0), finalFluxMat(2, 0)));
			//Vector3 mixedNodeVel = normalMatrix*Vector3(geometryNormal.dot(geometryVelocity), v1Normal.dot(v1Interp), v2Normal.dot(v2Interp));
			return mixedNodeVel;
			return Vector3();
		}
		#pragma endregion
	}
}