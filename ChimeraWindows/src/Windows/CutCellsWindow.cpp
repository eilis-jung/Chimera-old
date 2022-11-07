#include "Windows/CutCellsWindow.h"
#include "Visualization/MeshRenderer.h"

namespace Chimera {
	using namespace Rendering;

	namespace Windows {

		CutCellsWindow::CutCellsWindow(PolygonMeshRenderer<Vector2> *pPolyMeshRenderer, CutCellsVelocityRenderer2D<Vector2> *pVelRenderer)
			: BaseWindow(Vector2(16, 16), Vector2(300, 150), "CutCells Visualization"){
			m_pCutCells2D = pPolyMeshRenderer->getCutCells();;

			m_regularGridIndex = dimensions_t(-1, -1);
			m_cellDivergent = m_cellPressure = 0.0f;
			m_pCutCellsRenderer = pPolyMeshRenderer;
			m_pCutCellsVelRenderer = pVelRenderer;

			initializeAntTweakBar();
		}

		void CutCellsWindow::initializeAntTweakBar() {
			TwAddVarRW(m_pBaseBar, "drawSpecialCells", TW_TYPE_BOOL8, &m_pCutCellsRenderer->isDrawingCutCells(), " label='Draw Cells' help='Draw cut cells' group='Cells'");
			TwAddVarRW(m_pBaseBar, "drawSpecialCellsVertices", TW_TYPE_BOOL8, &m_pCutCellsRenderer->isDrawingVertices(), " label='Draw Cells Vertices' group='Cells'");
			TwAddVarRW(m_pBaseBar, "drawSelectedCells", TW_TYPE_INT32, &m_pCutCellsRenderer->getSelectedCell(), " label='Selected Cell Index' group='Cells'");
			TwAddVarRW(m_pBaseBar, "drawCellNeighbors", TW_TYPE_BOOL8, &m_drawCellNeighbors, "label='Draw Cell Neighbors' group='Cells'");

			//Geometry
			TwAddVarRW(m_pBaseBar, "drawEdgeNormals", TW_TYPE_BOOL8, &m_pCutCellsRenderer->isDrawingEdgeNormals(), "label='Draw Edge Normals' group='Cells Geometry'");
			
			TwEnumVal cutCellVelocityTypes[] = { { BaseWindow::vectorVisualization_t::drawVelocity, "Velocities" },
												 { BaseWindow::vectorVisualization_t::drawAuxiliaryVelocity, "Auxiliary Velocities" } };

			TwType velocityTypeAux = TwDefineEnum("CutCellVelocityTypeAux", cutCellVelocityTypes, 2);
			TwAddVarRW(m_pBaseBar, "cutCellVelType", velocityTypeAux, &m_pCutCellsVelRenderer->getVectorType(), "label='Velocity Type'  group='Velocity'");

			TwAddVarRW(m_pBaseBar, "drawEdgeVelocities", TW_TYPE_BOOL8, &m_pCutCellsVelRenderer->isDrawingEdgesVelocities(), "label='Draw Edge Velocities' group='Velocity'");
			TwAddVarRW(m_pBaseBar, "drawNodeVelocities", TW_TYPE_BOOL8, &m_pCutCellsVelRenderer->isDrawingNodalVelocities(), "label='Draw Nodal Velocities' group='Velocity'");
			TwAddVarRW(m_pBaseBar, "drawFGridVelocities", TW_TYPE_BOOL8, &m_pCutCellsVelRenderer->isDrawingFineGridVelocities(), "label='Draw Fine Grid Velocities' group='Velocity'");

			
			
			TwEnumVal fineGridVelTypes [] = {	{ CutCellsVelocityRenderer2D<Vector2>::drawCutCellsVelocitiesType_t::selectedCutCell, "Selected Cut-Cell"},
												{ CutCellsVelocityRenderer2D<Vector2>::drawCutCellsVelocitiesType_t::allCutCells, "All Cut-Cells"} };
			TwType velocityType = TwDefineEnum("CutCellVelocityType", fineGridVelTypes, 2);
			TwAddVarRW(m_pBaseBar, "drawFineGridVel", velocityType, &m_pCutCellsVelRenderer->getDrawingType(), "label='Velocity Drawing Mode'  group='Velocity'");

			//TwAddVarRW(m_pBaseBar, "velocityLenghtScale", TW_TYPE_FLOAT, &m_velocityLength, " label ='Velocity visualization length' group='Cut Cells' step=0.01 min='-0.15' max='0.15'");
		}

		/************************************************************************/
		/* Internal updates                                                     */
		/************************************************************************/

		void CutCellsWindow::updateAttributes() {
			//auto currCell = m_pCutCells2D->getCutCell(m_selectedCell);
			//m_regularGridIndex = currCell.getFace()->getGridCellLocation();
	/*		m_cellPressure = m_pCutCells2D->getPressure(m_selectedCell);
			m_cellDivergent = m_pCutCells2D->getDivergent(m_selectedCell);
			m_openEndedCell = m_pCutCells2D->getSpecialCell(m_selectedCell).isOpenEnded();*/
		}

		void CutCellsWindow::flushFaceAttributes() {
			for(map<string, bool>::iterator it = m_faceAttributesVars.begin(); it != m_faceAttributesVars.end(); it++) {
				TwRemoveVar(m_pBaseBar, it->first.c_str());
			}
			m_faceAttributesVars.clear();
		}
		void CutCellsWindow::updateFaceAttributes() {
			flushFaceAttributes();

			auto currCell = m_pCutCells2D->getCutCell(m_pCutCellsRenderer->getSelectedCell());

			for(int i = 0; i < currCell.getHalfEdges().size(); i++) {
				string faceAttribName =  "face" + intToStr(i);
				string label;
				auto pEdge = currCell.getHalfEdges()[i];
				switch (pEdge->getLocation()) {
					case rightHalfEdge:
						label = "Right Face";
					break;
					
					case leftHalfEdge:
						label = "Left Face";
					break;
					
					case bottomHalfEdge:
						label = "Bottom Face";
					break;
					
					case topHalfEdge:
						label = "Top Face";
					break;

					default:
						label = "Undefined face location";
					break;
				}
				string commandDesc = "label='" + label + "' group='Edge adjacencies'";
				/*if(pEdge->m_edgeNeighbors[0] != currCell.m_ID && pEdge->m_edgeNeighbors[0] != -1) {
					TwAddVarRO(m_pBaseBar, faceAttribName.c_str(), TW_TYPE_INT32, &pEdge->m_edgeNeighbors[0], commandDesc.c_str());
					m_faceAttributesVars[faceAttribName] = true;
				} else if(pEdge->m_edgeNeighbors[1] != -1){
					TwAddVarRO(m_pBaseBar, faceAttribName.c_str(), TW_TYPE_INT32, &pEdge->m_edgeNeighbors[1], commandDesc.c_str());
					m_faceAttributesVars[faceAttribName] = true;
				}*/
			}
		}

	}
}