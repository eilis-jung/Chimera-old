//  Copyright (c) 2013, Vinicius Costa Azevedo
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without
//	modification, are permitted provided that the following conditions are met: 
//
//1. Redistributions of source code must retain the above copyright notice, this
//	list of conditions and the following disclaimer. 
//	2. Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation
//	and/or other materials provided with the distribution. 
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
//	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//	The views and conclusions contained in the software and documentation are those
//	of the authors and should not be interpreted as representing official policies, 
//	either expressed or implied, of the FreeBSD Project.

#include "Windows/GridVisualizationWindow.h"
#include "RenderingUtils.h"
namespace Chimera {
	namespace Windows {

		#pragma region ButtonFunctions
		typedef struct saveFileData_t {
			BaseWindow::scalarVisualization_t * pScalarDrawingType;
			GridData2D * pGridData2D; 
		} saveFileData_t;
		
		void TW_CALL saveScalarFieldToFile(void *pClientData) {
			saveFileData_t *pSaveFileData = (saveFileData_t *) pClientData;
			Array2D<Scalar> scalarFieldArray = RenderingUtils::getInstance()->switchScalarField2D(*pSaveFileData->pScalarDrawingType, pSaveFileData->pGridData2D);

			string densityExportName("Flow Logs/scalarFieldSnapshot.log");
			auto_ptr<ofstream> fileStream(new ofstream(densityExportName.c_str()));
			
			//(*fileStream) << intToStr(scalarFieldArray.getDimensions().x) << " " << intToStr(scalarFieldArray.getDimensions().y) << endl;
			for(int i = 0; i < scalarFieldArray.getDimensions().x; i++) {
				for(int j = 0; j < scalarFieldArray.getDimensions().y; j++) {
					(*fileStream) << scalarToStr(scalarFieldArray(i, j)) << " ";
				}
				(*fileStream) << endl;
			}
		}
		#pragma endregion
		
		template <class VectorT>
		GridVisualizationWindow<VectorT>::GridVisualizationWindow(GridRenderer<VectorT> *pGridRenderer): BaseWindow(Vector2(16, 16), Vector2(300, 250), pGridRenderer->getGrid()->getGridName()) {
			m_pGridRenderer = pGridRenderer;
			m_drawGridPlaneXY = m_drawGridPlaneXZ = m_drawGridPlaneYZ = true;
			m_drawVelocitySlices = m_drawScalarFieldSlices = false;
			m_drawRegularCells = false;
			m_drawSolidCells = true;
			m_numOfGridPlanesDrawn = 1;
			m_gridDimensions = pGridRenderer->getGrid()->getDimensions();

			m_scalarDrawingType = drawPressure;
			m_vectorFieldDrawingType = drawVelocity;

			/** General */
			TwAddVarRW(m_pBaseBar, "gridRegularCells", TW_TYPE_BOOL8, &m_drawRegularCells, " label='Draw Grid Cells' help='Draw all regular cells' group='Regular Grid'");
			TwAddVarRW(m_pBaseBar, "gridSolidCells", TW_TYPE_BOOL8, &m_drawSolidCells, " label='Solid cells' help='Draw solid cells occupied by objects on the scenes' group='Regular Grid'");
			
			/** 3D grids only */
			if (m_gridDimensions.z != 0) {
				TwAddVarRW(m_pBaseBar, "drawSliceX", TW_TYPE_BOOL8, &m_drawGridPlaneYZ, " label='Draw YZ slice countour' group='Regular Grid' ");
				TwAddVarRW(m_pBaseBar, "drawSliceY", TW_TYPE_BOOL8, &m_drawGridPlaneXZ, " label='Draw XZ slice countour' group='Regular Grid' ");
				TwAddVarRW(m_pBaseBar, "drawSliceZ", TW_TYPE_BOOL8, &m_drawGridPlaneXY, " label='Draw XY slice countour' group='Regular Grid' ");

				TwAddVarRW(m_pBaseBar, "ithSliceX", TW_TYPE_INT32, &m_ithRenderingPlanes.x, " label='YZ Slice index' group='Regular Grid' ");
				TwAddVarRW(m_pBaseBar, "ithSliceY", TW_TYPE_INT32, &m_ithRenderingPlanes.y, " label='XZ Slice index' group='Regular Grid' ");
				TwAddVarRW(m_pBaseBar, "ithSliceZ", TW_TYPE_INT32, &m_ithRenderingPlanes.z, " label='XY Slice index' group='Regular Grid' ");

				TwAddVarRW(m_pBaseBar, "numberOfDrawnPlanes", TW_TYPE_INT32, &m_numOfGridPlanesDrawn, " label='Number of planes drawn' group='Regular Grid' ");
				//string defName = "'" + m_windowName + "'/'Regular Grid' opened=false";
				//TwDefine(defName.c_str()); // fold the group 'Regular Grid'
				string defName = "'" + m_windowName + "' size='300 400'";
				TwDefine(defName.c_str()); // Resize window
			}

			/** Group velocity field*/
			TwEnumVal vectorFields[] = {	{drawNoVectorField, "None"}, {drawStaggeredVelocity, "Staggered Velocity"}, 
											{drawVelocity, "Velocity"}, {drawAuxiliaryVelocity, "Intermediate Velocity"},
											{drawNodeVelocity, "Node Velocity"}, {drawGradients, "Scalar Gradients"} };
			TwType vectorType = TwDefineEnum("VelocityType", vectorFields, 6);
			TwAddVarRW(m_pBaseBar, "drawVelocityFieldType", vectorType, &m_vectorFieldDrawingType, "label='Type'  group='Velocity Field'");
			TwAddVarRW(m_pBaseBar, "velocityScale", TW_TYPE_FLOAT, (void *)&pGridRenderer->getVectorFieldRenderer().velocityLength, "label='Scale'  group='Velocity Field' step=0.01 min='-0.15' max='0.15'");
			TwAddVarRW(m_pBaseBar, "drawFineGridVelocities", TW_TYPE_BOOL8, &pGridRenderer->getVectorFieldRenderer().m_drawFineGridVelocities, " label='Draw Fine grid Velocities' group='Velocity Field' ");
			if (m_gridDimensions.z != 0) {
				TwAddVarRW(m_pBaseBar, "drawVelocityFieldSlices", TW_TYPE_BOOL8, &m_drawVelocitySlices, " label='Draw Velocity Slice' group='Velocity Field' ");
			}
			//string defName = "'" + m_windowName + "'/'Velocity Field' group='Regular Grid'";
			//TwDefine(defName.c_str());

			/** Group scalar field*/
			TwEnumVal scalarEV[] = { { drawNoScalarField, "None" }, { drawVorticity, "Vorticity" }, { drawLevelSet, "Level Set" }, { drawPressure, "Pressure" }, { drawDivergent, "Divergent" },
									 { drawStreamfunction, "Streamfunction"},
									 { drawDensityField, "Density" }, { drawTemperature, "Temperature" }, { drawFineGridScalars, "Fine-grid Scalar Field" },
									 { drawKineticEnergy, "Kinetic Energy" },{ drawKineticEnergyChange, "Kinetic Energy Change" } };
			TwType scalarType = TwDefineEnum("ScalarType", scalarEV, 11);
			if (m_gridDimensions.z != 0) {
				TwAddVarRW(m_pBaseBar, "drawScalarFieldSlices", TW_TYPE_BOOL8, &m_drawScalarFieldSlices, " label='Draw Scalar Fields' group='Scalar Field' ");
			}
			TwAddVarRW(m_pBaseBar, "drawScalarFieldType", scalarType, &m_scalarDrawingType, "label='Type'  group='Scalar Field'");
			TwAddVarRW(m_pBaseBar, "minScalarFieldVal", TW_TYPE_FLOAT, (void *)&pGridRenderer->getScalarFieldRenderer().m_minScalarFieldVal, "label='Minimum Value'  group='Scalar Field'");
			TwAddVarRW(m_pBaseBar, "maxScalarFieldVal", TW_TYPE_FLOAT, (void *)&pGridRenderer->getScalarFieldRenderer().m_maxScalarFieldVal, "label='Maximum Value'  group='Scalar Field'");
			TwAddVarRW(m_pBaseBar, "updateMinMax", TW_TYPE_BOOL8, &pGridRenderer->getScalarFieldRenderer().m_updateScalarMinMax, "label='Update min/max'  group='Scalar Field'");
			/** Export Scalar field */
			saveFileData_t *pSaveFileData = new saveFileData_t();;
			pSaveFileData->pScalarDrawingType = &m_scalarDrawingType;
			pSaveFileData->pGridData2D = pGridRenderer->getGrid()->getGridData2D();
			TwAddButton(m_pBaseBar, "exportScalarField", saveScalarFieldToFile, pSaveFileData, " label='Export to logfile' group='Scalar Field' ");
			/** Field coloring scheme*/
			TwEnumVal coloringTypeEV[] = { { Rendering::viridis, "Viridis" }, { Rendering::jet, "Jet" }, { Rendering::grayscale, "GrayScale" } };
			TwType coloringType = TwDefineEnum("ColoringType", coloringTypeEV, 3);
			TwAddVarRW(m_pBaseBar, "coloringScheme", coloringType, (void *)&m_pGridRenderer->getScalarFieldRenderer().m_colorScheme, "label='Coloring scheme'  group='Scalar Field'");

			//defName = "'" + m_windowName + "'/'Scalar Field' group='Regular Grid'";
			//TwDefine(defName.c_str());
			//int opened = 0;
			//TwSetParam(m_pBaseBar, "Options", "opened", TW_PARAM_INT32, 1, &opened);
		}
		
		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		template <class VectorT>
		void GridVisualizationWindow<VectorT>::update() {
			m_pGridRenderer->getScalarFieldRenderer().getIsocontourRenderer().getParams().currentScalarVisualization = m_scalarDrawingType;
			m_pGridRenderer->getScalarFieldRenderer().getIsocontourRenderer().update();
			dimensions_t gridDimensions = m_pGridRenderer->getGrid()->getDimensions();
			m_ithRenderingPlanes.x = clamp(m_ithRenderingPlanes.x, 0, gridDimensions.x);
			m_ithRenderingPlanes.y = clamp(m_ithRenderingPlanes.y, 0, gridDimensions.y);
			m_ithRenderingPlanes.y = clamp(m_ithRenderingPlanes.z, 0, gridDimensions.z);
		}
		template <class VectorT>
		void GridVisualizationWindow<VectorT>::addParticleSystem(ParticleSystem2D *pParticleSystem) {
			m_pParticleSystem = pParticleSystem;
			TwAddVarRW(m_pBaseBar, "drawParticles", TW_TYPE_BOOL8, &m_pParticleSystem->getRenderingParams().m_draw, " label='Draw (particles + trails)' group='Particles'");
			TwAddVarRW(m_pBaseBar, "drawParticles2", TW_TYPE_BOOL8, &m_pParticleSystem->getRenderingParams().m_drawParticles, " label='Draw Particles' group='Particles'");
			TwAddVarRW(m_pBaseBar, "drawVelocities", TW_TYPE_BOOL8, &m_pParticleSystem->getRenderingParams().m_drawVelocities, " label='Draw Particle Velocities' group='Particles'");
			TwAddVarRW(m_pBaseBar, "particleVelocityScale", TW_TYPE_FLOAT, &m_pParticleSystem->getRenderingParams().velocityScale, " label='Velocity Scale' group='Particles'");
			TwAddVarRW(m_pBaseBar, "particleSize", TW_TYPE_FLOAT, &m_pParticleSystem->getRenderingParams().particleSize, " label='Particles size' group='Particles'");
			string maxTrailSizeStr = string("label='Particles Trail Size' group='Particles' min='0' max='") + intToStr(ParticleSystem2D::s_maxTrailSize - 1) + string("'");
			TwAddVarRW(m_pBaseBar, "trailSize", TW_TYPE_INT32, &m_pParticleSystem->getRenderingParams().trailSize, maxTrailSizeStr.c_str());
			
			TwEnumVal coloringTypeEV[] = { {Rendering::singleColor, "Single color"}, {Rendering::jet, "Jet (Scalar Field)"}, {Rendering::grayscale, "GrayScale (Scalar Field)"} };
			TwType coloringType = TwDefineEnum("particlesColoringType", coloringTypeEV, 3);
			TwAddVarRW(m_pBaseBar, "particleColor", TW_TYPE_COLOR3F, (void *) m_pParticleSystem->getRenderingParams().particleColor,  "label='Particles color'  group='Particles'");
			TwAddVarRW(m_pBaseBar, "particlesColoringScheme", coloringType, (void *)&m_pParticleSystem->getRenderingParams().colorScheme, "label='Particles coloring scheme'  group='Particles'");
			TwAddVarRW(m_pBaseBar, "clampParticlesTrails", TW_TYPE_BOOL8, &m_pParticleSystem->getRenderingParams().m_clipParticlesTrails, " label='Clamp particles trails' group='Particles'");
		}

		template <class VectorT>
		void GridVisualizationWindow<VectorT>::addParticleSystem(ParticleSystem3D *pParticleSystem) {
			TwAddVarRW(m_pBaseBar, "drawParticles", TW_TYPE_BOOL8, &pParticleSystem->getRenderingParams().m_draw, " label='Draw Particles' group='Particles'");
			TwAddVarRW(m_pBaseBar, "particleSize", TW_TYPE_FLOAT, &pParticleSystem->getRenderingParams().particleSize, " label='Particles size' group='Particles'");
			string maxTrailSizeStr = string("label='Particles Trail Size' group='Particles' min='0' max='") + intToStr(ParticleSystem2D::s_maxTrailSize - 1) + string("'");
			TwAddVarRW(m_pBaseBar, "trailSize", TW_TYPE_INT32, &pParticleSystem->getRenderingParams().trailSize, maxTrailSizeStr.c_str());
			TwAddVarRW(m_pBaseBar, "drawParticlesVelocities", TW_TYPE_BOOL8, &pParticleSystem->getRenderingParams().drawVelocities, " label='Draw Velocities' group='Particles'");
			TwAddVarRW(m_pBaseBar, "drawParticlesNormals", TW_TYPE_BOOL8, &pParticleSystem->getRenderingParams().drawNormals, " label='Draw Normals' group='Particles'");

			TwEnumVal coloringTypeEV[] = { {Rendering::singleColor, "Single color"}, {Rendering::jet, "Jet (Scalar Field)"}, {Rendering::grayscale, "GrayScale (Scalar Field)"} };
			TwType coloringType = TwDefineEnum("particlesColoringType", coloringTypeEV, 3);
			TwAddVarRW(m_pBaseBar, "particleColor", TW_TYPE_COLOR3F, (void *) pParticleSystem->getRenderingParams().particleColor,  "label='Particles color'  group='Particles'");
			TwAddVarRW(m_pBaseBar, "particlesColoringScheme", coloringType, (void *)&pParticleSystem->getRenderingParams().colorScheme, "label='Particles coloring scheme'  group='Particles'");
			TwAddVarRW(m_pBaseBar, "selectedDimParticle", TW_TYPE_BOOL8, &pParticleSystem->getRenderingParams().drawSelectedVoxelParticles, "label='Select particles by voxel'  group='Particles'");

			TwAddVarRW(m_pBaseBar, "selectParticleGridSlice", TW_TYPE_BOOL8, &pParticleSystem->getRenderingParams().drawParticlePerSlice, "label='Select particles by grid slice'  group='Particles'");
			TwAddVarRW(m_pBaseBar, "particleGridSlice", TW_TYPE_INT32, &pParticleSystem->getRenderingParams().gridSliceDimension, "Ith grid slice'  group='Particles'");
		}
		/************************************************************************/
		/* GridVisualization declarations - Linking time                        */
		/************************************************************************/
		template GridVisualizationWindow<Vector2>;
		template GridVisualizationWindow<Vector3>;

	}
}