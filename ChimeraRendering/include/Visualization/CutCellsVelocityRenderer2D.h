#pragma once

#ifndef _CHIMERA_CUTCELLS_VELOCITY_RENDERER_
#define _CHIMERA_CUTCELLS_VELOCITY_RENDERER_
#pragma once

#include "ChimeraResources.h"
#include "ChimeraRenderingCore.h"
#include "RenderingUtils.h"
#include "ChimeraCutCells.h"
#include "Windows/BaseWindow.h"

namespace Chimera {
	using namespace CutCells;

	namespace Rendering {


		template <class VectorType> 
		class CutCellsVelocityRenderer2D {

		public:
			#pragma region ExternalStructures
			typedef enum drawCutCellsVelocitiesType_t {
				allCutCells,
				selectedCutCell
			} drawCutCellsVelocitiesType_t;
			#pragma endregion

			#pragma region Constructors
			CutCellsVelocityRenderer2D(CutCells2D<VectorType> *pCutCells, const int &selectedCutCellIndex) : m_selectedCutCell(selectedCutCellIndex) {
				m_pCutCells = pCutCells;
				m_drawEdgeVelocities = m_drawNodalVelocities = m_drawFineGridVelocities = false;
				m_drawVelocitiesType = selectedCutCell;
				m_mainVelocityType = BaseWindow::drawVelocity;
			}
			#pragma endregion

			#pragma region Functionalities
			void draw();
			#pragma endregion

			#pragma region AccessFunctions
			bool & isDrawingEdgesVelocities() {
				return m_drawEdgeVelocities;
			}
			bool & isDrawingNodalVelocities() {
				return m_drawNodalVelocities;
			}
			bool & isDrawingFineGridVelocities() {
				return m_drawFineGridVelocities;
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
			bool m_drawEdgeVelocities;
			bool m_drawNodalVelocities;
			bool m_drawFineGridVelocities;
			drawCutCellsVelocitiesType_t m_drawVelocitiesType;
			BaseWindow::vectorVisualization_t m_mainVelocityType;
			CutCells2D<VectorType> *m_pCutCells;
			const int & m_selectedCutCell;
			#pragma endregion

			#pragma region PrivateDrawingFunctions
			void drawEdgeVelocities();
			void drawEdgeVelocities(uint edgeID);
			void drawNodalVelocities();
			void drawNodalVelocities(uint cellID);
			void drawFineGridVelocities();
			#pragma endregion
		};
	}
}

#endif