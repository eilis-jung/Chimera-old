#include "Kernels/TransferKernel.h"

namespace Chimera {
	namespace Advection {
		
		#pragma region Functionalities
		template <>
		void TransferKernel<Vector2>::getCellList(const Vector2 &position, vector<dimensions_t> &cellsList) {
			Scalar dx = m_pGridData->getGridSpacing();
			Vector2 gridSpacePos = position / dx;
			
			cellsList.clear();

			/** Maximum number of the cells spanned by the kernel: ceil guarantees the minimum of 1 cell */
			int numCellsSpan = ceil(m_kernelSize/dx) - 1;
			dimensions_t currCellDim(floor(gridSpacePos.x), floor(gridSpacePos.y));

			for (int i = currCellDim.x - numCellsSpan; i <= currCellDim.x + numCellsSpan; i++) {
				for (int j = currCellDim.y - numCellsSpan; j <= currCellDim.y + numCellsSpan; j++) {
					
					if (i < 0 || i > m_pGridData->getDimensions().x - 1 ||
						j < 0 || j > m_pGridData->getDimensions().y - 1) {
						continue;
					}
					Vector2 currCellPosition((i + 0.5)*dx, (j + 0.5)*dx);
					if ((position - currCellPosition).length() < m_kernelSize) {
						cellsList.push_back(dimensions_t(i, j));
					}
				}
			}
		}
		template <>
		void TransferKernel<Vector3>::getCellList(const Vector3 &position, vector<dimensions_t> &cellsList) {
			Scalar dx = m_pGridData->getGridSpacing();
			Vector3 gridSpacePos = position / dx;

			cellsList.clear();

			/** Maximum number of the cells spanned by the kernel: ceil guarantees the minimum of 1 cell */
			int numCellsSpan = ceil(m_kernelSize / dx) - 1;
			dimensions_t currCellDim(floor(gridSpacePos.x), floor(gridSpacePos.y), floor(gridSpacePos.z));

			for (int i = currCellDim.x - numCellsSpan; i <= currCellDim.x + numCellsSpan; i++) {
				for (int j = currCellDim.y - numCellsSpan; j <= currCellDim.y + numCellsSpan; j++) {
					for (int k = currCellDim.z - numCellsSpan; k <= currCellDim.z + numCellsSpan; k++) {

						if (i < 0 || i > m_pGridData->getDimensions().x - 1 ||
							j < 0 || j > m_pGridData->getDimensions().y - 1 ||
							k < 0 || k > m_pGridData->getDimensions().z - 1) {
							continue;
						}
						Vector3 currCellPosition((i + 0.5)*dx, (j + 0.5)*dx, (k + 0.5)*dx);
						if ((position - currCellPosition).length() < m_kernelSize) {
							cellsList.push_back(dimensions_t(i, j, k));
						}
					}
				}
			}
		}

		template <>
		void TransferKernel<Vector2>::getCutCellList(const Vector2 &position, vector<int> &cutCellsIDList) {

		}

		template <>
		void TransferKernel<Vector3>::getCutCellList(const Vector3 &position, vector<int> &cutCellsIDList) {
		
		}

		template TransferKernel<Vector2>;
		template TransferKernel<Vector3>;
		#pragma endregion
	}
}