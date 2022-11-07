#include "Visualization/CutVoxelsVelocityRenderer3D.h"

namespace Chimera {
	namespace Rendering {

		#pragma region Functionalities
		template<class VectorType>
		void CutVoxelsVelocityRenderer3D<VectorType>::draw() {
			if (m_drawFaceVelocities)
				drawFaceVelocities();

			if (m_drawNodalVelocities)
				drawNodalVelocities();

			if (m_drawFineGridVelocities)
				drawFineGridVelocities();
		}

		template<class VectorType>
		void CutVoxelsVelocityRenderer3D<VectorType>::update() {
			updateFineGridVelocities();
			/*if (m_drawFineGridVelocities) {
				if (m_lastFineSubdivis != m_currFineSubdivis) {
					updateFineGridVelocities();
					m_lastFineSubdivis = m_currFineSubdivis;
				}
			}*/
		}
		#pragma endregion


		#pragma region PrivateDrawingFunctions
		template<class VectorType>
		void CutVoxelsVelocityRenderer3D<VectorType>::drawFaceVelocities() {
			if (m_drawVelocitiesType == selectedCutCell) {
				drawFaceVelocities(m_selectedCutCell);
			}
			else {
				for (int i = 0; i < m_pCutVoxels->getNumberCutVoxels(); i++) {
					drawFaceVelocities(i);
				}
			}
		}
		template<class VectorType>
		void CutVoxelsVelocityRenderer3D<VectorType>::drawFaceVelocities(uint cellIndex) {
			auto cutVoxel = m_pCutVoxels->getCutVoxel(cellIndex);
			auto halfFaces = cutVoxel.getHalfFaces();
			for (uint i = 0; i < halfFaces.size(); i++) {
				VectorType initialPoint = halfFaces[i]->getCentroid();
				VectorType finalPoint;

				if (m_mainVelocityType == BaseWindow::drawVelocity)
					finalPoint = halfFaces[i]->getFace()->getVelocity();
				else if (m_mainVelocityType == BaseWindow::drawAuxiliaryVelocity)
					finalPoint = halfFaces[i]->getFace()->getAuxiliaryVelocity();
				
				RenderingUtils::getInstance()->drawVector(initialPoint, initialPoint + finalPoint*m_velScaleLength);
			}
		} 

		template<class VectorType>
		void CutVoxelsVelocityRenderer3D<VectorType>::drawNodalVelocities() {
			if (m_drawVelocitiesType == selectedCutCell) {
				drawNodalVelocities(m_selectedCutCell);
			}
			else {
				for (int i = 0; i < m_pCutVoxels->getNumberCutVoxels(); i++) {
					drawNodalVelocities(i);
				}
			}	
		}

		template<class VectorType>
		void CutVoxelsVelocityRenderer3D<VectorType>::drawNodalVelocities(uint cellIndex) {
			auto cutVoxel = m_pCutVoxels->getCutVoxel(cellIndex);
			for (auto iter = cutVoxel.getVerticesMap().begin(); iter != cutVoxel.getVerticesMap().end(); iter++) {
				VectorType initialPoint = iter->second->getPosition();
				VectorType finalPoint;

				if(m_mainVelocityType == BaseWindow::drawVelocity)
					finalPoint = iter->second->getVelocity();
				else if(m_mainVelocityType == BaseWindow::drawAuxiliaryVelocity)
					finalPoint = iter->second->getAuxiliaryVelocity();

				RenderingUtils::getInstance()->drawVector(initialPoint, initialPoint + finalPoint*m_velScaleLength);
			}
		}

		template<class VectorType>
		void CutVoxelsVelocityRenderer3D<VectorType>::drawFineGridVelocities() {
			for (int i = 0; i < m_fineGridVelocities.size(); i++) {
				RenderingUtils::getInstance()->drawVector(m_fineGridVelocities[i].first, m_fineGridVelocities[i].first + m_fineGridVelocities[i].second*m_velScaleLength);
			}
		}
		#pragma endregion

		#pragma region PrivateFunctionalities
		template<class VectorType>
		void CutVoxelsVelocityRenderer3D<VectorType>::updateFineGridVelocities() {
			if (m_pCutVelocitiesInterpolant && m_selectedCutCell != -1) {
				m_fineGridVelocities.clear();
				Scalar dx = m_pCutVoxels->getGridSpacing();
				Scalar finedx = dx/(m_currFineSubdivis + 1);
				auto cutVoxel = m_pCutVoxels->getCutVoxel(m_selectedCutCell);
				const dimensions_t &voxelLocation = cutVoxel.getVolume()->getGridCellLocation();
				for (int i = 1; i <= m_currFineSubdivis; i++) {
					for (int j = 1; j <= m_currFineSubdivis; j++) {
						for (int k = 1; k <= m_currFineSubdivis; k++) {
							VectorType position(voxelLocation.x*dx + i*finedx, voxelLocation.y*dx + j*finedx, voxelLocation.z*dx + k*finedx);
							VectorType velocity = m_pCutVelocitiesInterpolant->interpolate(Vector3(position.x, position.y, position.z));
							m_fineGridVelocities.push_back(pair<VectorType, VectorType>(position, velocity));
						}
					}
				}
			}
		}
		#pragma endregion


		template class CutVoxelsVelocityRenderer3D<Vector3>;
		template class CutVoxelsVelocityRenderer3D<Vector3D>;
	}
}