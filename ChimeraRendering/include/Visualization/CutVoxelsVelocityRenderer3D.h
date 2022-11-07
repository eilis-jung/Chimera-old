#pragma once

#ifndef _CHIMERA_CUTVOXELS_VELOCITY_RENDERER_
#define _CHIMERA_CUTVOXELS_VELOCITY_RENDERER_
#pragma once

#include "ChimeraResources.h"
#include "ChimeraRenderingCore.h"
#include "RenderingUtils.h"
#include "ChimeraCutCells.h"
#include "Windows/BaseWindow.h"
#include "ChimeraInterpolation.h"

namespace Chimera {
	using namespace CutCells;
	using namespace Interpolation;
	namespace Rendering {


		template <class VectorType> 
		class CutVoxelsVelocityRenderer3D {

		public:
			#pragma region ExternalStructures
			typedef enum drawCutCellsVelocitiesType_t {
				allCutCells,
				selectedCutCell
			} drawCutCellsVelocitiesType_t;
			#pragma endregion

			#pragma region Constructors
			CutVoxelsVelocityRenderer3D(CutVoxels3D<VectorType> *pCutVoxels, const int &selectedCutCellIndex, 
										MeanValueInterpolant3D<VectorType> *pVelInterpolant) : m_selectedCutCell(selectedCutCellIndex) {
				m_pCutVoxels = pCutVoxels;
				m_pCutVelocitiesInterpolant = pVelInterpolant;
				m_drawFaceVelocities = m_drawNodalVelocities = m_drawFineGridVelocities = false;
				m_drawVelocitiesType = selectedCutCell;
				m_mainVelocityType = BaseWindow::drawVelocity;
				m_velScaleLength = 0.01;
				m_lastFineSubdivis = m_currFineSubdivis = 0;
			}
			#pragma endregion

			#pragma region Functionalities
			void draw();

			void update();
			#pragma endregion

			#pragma region AccessFunctions
			bool & isDrawingFaceVelocities() {
				return m_drawFaceVelocities;
			}
			bool & isDrawingNodalVelocities() {
				return m_drawNodalVelocities;
			}
			bool & isDrawingFineGridVelocities() {
				return m_drawFineGridVelocities;
			}

			Scalar &getVelocityScaleLength() {
				return m_velScaleLength;
			}

			uint & getFineGridVelocitiesSubdivisions() {
				return m_currFineSubdivis;
			}
			drawCutCellsVelocitiesType_t & getDrawingType() {
				return m_drawVelocitiesType;
			}

			BaseWindow::vectorVisualization_t & getVectorType() {
				return m_mainVelocityType;
			}

			#pragma endregion
		protected:

			#pragma region ClassMembers
			bool m_drawFaceVelocities;
			bool m_drawNodalVelocities;
			bool m_drawFineGridVelocities;

			drawCutCellsVelocitiesType_t m_drawVelocitiesType;
			BaseWindow::vectorVisualization_t m_mainVelocityType;
			Scalar m_velScaleLength;
			uint m_lastFineSubdivis;
			uint m_currFineSubdivis;
			vector<pair<VectorType, VectorType>> m_fineGridVelocities;

			MeanValueInterpolant3D<VectorType> *m_pCutVelocitiesInterpolant;
			CutVoxels3D<VectorType> *m_pCutVoxels;
			const int & m_selectedCutCell;
			#pragma endregion

			#pragma region PrivateDrawingFunctions
			void drawFaceVelocities();
			void drawFaceVelocities(uint edgeID);
			void drawNodalVelocities();
			void drawNodalVelocities(uint cellID);
			void drawFineGridVelocities();
			#pragma endregion

			#pragma region PrivateFunctionalities
			void updateFineGridVelocities();
			#pragma endregion
		};
	}
}

#endif