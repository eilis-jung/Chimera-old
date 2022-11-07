#include "Windows/CutVoxelsWindow.h"
#include "Visualization/MeshRenderer.h"

namespace Chimera {
	using namespace Rendering;

	namespace Windows {

		#pragma region Constructors
		template <class VectorType>
		CutVoxelsWindow<VectorType>::CutVoxelsWindow(Rendering::VolumeMeshRenderer<VectorType> *pVolumeRenderer, Rendering::CutVoxelsVelocityRenderer3D<VectorType> *pVelRenderer)
			: BaseWindow(Vector2(16, 16), Vector2(300, 300), "CutCells Visualization") {
			m_pCutVoxels = pVolumeRenderer->getCutVoxels();
			m_pCutVoxelsRenderer = pVolumeRenderer;
			m_pCutVelocitiesRenderer = pVelRenderer;

			m_regularGridIndex = dimensions_t(-1, -1, -1);
			m_cellDivergent = m_cellPressure = 0.0f;
			m_drawVoxelsNeighbors = false;

			initializeAntTweakBar();
		}
		#pragma endregion

		#pragma region Initialization
		template <class VectorType>
		void CutVoxelsWindow<VectorType>::initializeAntTweakBar() {

			TwAddVarRW(m_pBaseBar, "drawCutVoxels", TW_TYPE_BOOL8, &m_pCutVoxelsRenderer->isDrawingCutVoxels(), " label='Draw Voxels' help='Draw cut voxels' group='Voxels'");
			TwAddVarRW(m_pBaseBar, "drawVoxelsVertices", TW_TYPE_BOOL8, &m_pCutVoxelsRenderer->isDrawingVertices(), " label='Draw Voxels Vertices' group='Voxels'");
			TwAddVarRW(m_pBaseBar, "isDrawingSelectedCutVoxel", TW_TYPE_BOOL8, &m_pCutVoxelsRenderer->isDrawingSelectedCutVoxel(), " label='Draw Selected Cut-Voxel' group='Voxels'");
			TwAddVarRW(m_pBaseBar, "selectedCutVoxelIndex", TW_TYPE_INT32, &m_pCutVoxelsRenderer->getSelectedCutVoxel(), " label='Selected Cut-Voxel Index' group='Voxels'");
			TwAddVarRW(m_pBaseBar, "drawCutVoxelsNeighbors", TW_TYPE_BOOL8, &m_pCutVoxelsRenderer->isDrawingNeighbors(), "label='Draw Voxels Neighbors' group='Voxels'");

			TwAddVarRW(m_pBaseBar, "drawVoxelsNormals", TW_TYPE_BOOL8, &m_pCutVoxelsRenderer->isDrawingCutVoxelNormals(), " label='Draw Voxels Normals' group='Selected Voxel'");
			TwAddVarRW(m_pBaseBar, "drawVoxelsFacesCentroids", TW_TYPE_BOOL8, &m_pCutVoxelsRenderer->isDrawingCutFacesCentroids(), " label='Draw Faces Centroids' group='Selected Voxel'");

			TwAddVarRW(m_pBaseBar, "drawXYLineMeshes", TW_TYPE_BOOL8, &m_pCutVoxelsRenderer->getLineMeshRenderers()[0]->isDrawing(), " label='Draw XY Line Meshes' group='Mesh'");
			TwAddVarRW(m_pBaseBar, "drawXZLineMeshes", TW_TYPE_BOOL8, &m_pCutVoxelsRenderer->getLineMeshRenderers()[1]->isDrawing(), " label='Draw XZ Line Meshes' group='Mesh'");
			TwAddVarRW(m_pBaseBar, "drawYZLineMeshes", TW_TYPE_BOOL8, &m_pCutVoxelsRenderer->getLineMeshRenderers()[2]->isDrawing(), " label='Draw YZ Line Meshes' group='Mesh'");

			string defName = "'" + m_windowName + "'/'Mesh' opened=false";
			TwDefine(defName.c_str()); // fold the group 'Mesh'
	
			//Geometry
			//TwAddVarRW(m_pBaseBar, "drawEdgeNormals", TW_TYPE_BOOL8, &m_pCutCellsRenderer->isDrawingEdgeNormals(), "label='Draw Edge Normals' group='Voxels Geometry'");
			
			TwEnumVal cutCellVelocityTypes[] = { { BaseWindow::vectorVisualization_t::drawVelocity, "Velocities" },
												 { BaseWindow::vectorVisualization_t::drawAuxiliaryVelocity, "Auxiliary Velocities" } };

			//Not drawing velocities, for now
			TwType velocityTypeAux = TwDefineEnum("CutCellVelocityTypeAux", cutCellVelocityTypes, 2);
			TwAddVarRW(m_pBaseBar, "cutVoxelVelType", velocityTypeAux, &m_pCutVelocitiesRenderer->getVectorType(), "label='Velocity Type'  group='Velocity'");
			TwAddVarRW(m_pBaseBar, "drawFaceVelocities", TW_TYPE_BOOL8, &m_pCutVelocitiesRenderer->isDrawingFaceVelocities(), "label='Draw Face Velocities' group='Velocity'");
			TwAddVarRW(m_pBaseBar, "drawNodeVelocities", TW_TYPE_BOOL8, &m_pCutVelocitiesRenderer->isDrawingNodalVelocities(), "label='Draw Nodal Velocities' group='Velocity'");
			
			TwEnumVal cutVoxelsCatType [] = {	{ CutVoxelsVelocityRenderer3D<Vector2>::drawCutCellsVelocitiesType_t::selectedCutCell, "Selected Cut-Cell"},
												{ CutVoxelsVelocityRenderer3D<Vector2>::drawCutCellsVelocitiesType_t::allCutCells, "All Cut-Cells"} };
			TwType velocityType = TwDefineEnum("CutVoxelCatType", cutVoxelsCatType, 2);
			TwAddVarRW(m_pBaseBar, "drawFineGridVel", velocityType, &m_pCutVelocitiesRenderer->getDrawingType(), "label='Velocity Drawing Mode'  group='Velocity'");

			TwAddVarRW(m_pBaseBar, "drawFGridVelocities", TW_TYPE_BOOL8, &m_pCutVelocitiesRenderer->isDrawingFineGridVelocities(), "label='Draw Fine Grid Velocities' group='Velocity'");
			TwAddVarRW(m_pBaseBar, "drawFGridVelocitiesSubdivis", TW_TYPE_UINT32, &m_pCutVelocitiesRenderer->getFineGridVelocitiesSubdivisions(), "label='Fine Grid Subdivision' group='Velocity'");


			TwAddVarRW(m_pBaseBar, "velocityLenghtScale", TW_TYPE_FLOAT, &m_pCutVelocitiesRenderer->getVelocityScaleLength(), " label ='Velocity visualization length' group='Velocity' step=0.01 min='-0.5' max='0.5'");

			TwAddVarRW(m_pBaseBar, "selectedNode", TW_TYPE_UINT32, &m_pCutVoxelsRenderer->getSelectedNode(), "label = 'Selected Node' group = 'Nodes'");
			TwAddVarRW(m_pBaseBar, "drawMixedNodeNeighbors", TW_TYPE_BOOL8, &m_pCutVoxelsRenderer->isDrawingMixedNodeNeighbors(), "label = 'Draw Mixed Node Neighbors' group = 'Nodes'");

		}

		#pragma region PrivateFunctionalities
		template <class VectorType>
		void CutVoxelsWindow<VectorType>::updateAttributes() {
			//auto currCell = m_pCutCells2D->getCutCell(m_selectedCell);
			//m_regularGridIndex = currCell.getFace()->getGridCellLocation();
	/*		m_cellPressure = m_pCutCells2D->getPressure(m_selectedCell);
			m_cellDivergent = m_pCutCells2D->getDivergent(m_selectedCell);
			m_openEndedCell = m_pCutCells2D->getSpecialCell(m_selectedCell).isOpenEnded();*/
		}
		#pragma endregion


		template class CutVoxelsWindow<Vector3>;
		template class CutVoxelsWindow<Vector3D>;
		
	}
}