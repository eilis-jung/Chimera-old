#include "Visualization/IsocontourRenderer.h"
#include "RenderingUtils.h"
namespace Chimera{
	namespace Rendering {

		#pragma region ConstructorsDestructors
		IsocontourRenderer::IsocontourRenderer(GridData2D *pGridData) {
			m_pGridData = pGridData;
			initializeVBOs();
			generateIsolines();
			updateVBOs();
		}
		#pragma endregion ConstructorsDestructors
		#pragma region VBOsFunctions
		void IsocontourRenderer::initializeVBOs() {
			m_pPointsVBO = new GLuint();
			glGenBuffers(1, m_pPointsVBO);
		}
		void IsocontourRenderer::updateVBOs() {
			glBindBuffer(GL_ARRAY_BUFFER, *m_pPointsVBO);
			int totalSize = 0;
			for(int i = 0; i < m_isoLines.size(); i++) {
				totalSize += m_isoLines[i].size();
			}
			if(totalSize > 0)
				glBufferData(GL_ARRAY_BUFFER, totalSize*sizeof(Vector2), &m_isoLines[0], GL_DYNAMIC_DRAW);
		}
		#pragma endregion VBOsFunctions

		#pragma region DrawingFunctions
		void IsocontourRenderer::drawIsocontours() const {
			if(m_params.drawIsocontours) {
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				glBindBuffer(GL_ARRAY_BUFFER, *m_pPointsVBO);
				glVertexPointer(2, GL_FLOAT, 0, 0);
				int totalLinesRendered = 0;
				for(int i = 0; i < m_isoLines.size(); i++) {
					glDrawArrays(GL_POLYGON, totalLinesRendered, m_isoLines[i].size());
					totalLinesRendered += m_isoLines[i].size();
				}
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			}
			if(m_params.drawIsopoints) {
				glBegin(GL_POINTS);
				for(int i = 0; i < m_isoLines.size(); i++) {
					for(int j = 0; j < m_isoLines[i].size(); j++) {
						glVertex2f(m_isoLines[i][j].x, m_isoLines[i][j].y);
					}
				}
				glEnd();
			}
		}
 		#pragma endregion DrawingFunctions

		#pragma region UpdateFunctions
		void IsocontourRenderer::update() {
			if(m_oldParams.initialIsoValue != m_params.initialIsoValue ||
				m_oldParams.isoStepHorizontal != m_params.isoStepHorizontal ||
				m_oldParams.isoStepVertical != m_params.isoStepVertical ||
				m_oldParams.numIsocontours != m_params.numIsocontours) {
				for(int i = 0; i < m_isoLines.size(); i++) {
					m_isoLines[i].clear();
				}
				generateIsolines();
				updateVBOs();
			}
			m_oldParams = m_params;
 		}
		#pragma endregion UpdateFunctions


		#pragma region AuxiliaryFunctions
		void IsocontourRenderer::generateIsolines() {
			if(m_params.currentScalarVisualization != BaseWindow::drawNoScalarField) {
				vector<Vector2> isocontour;
				for(int i = 0; i < m_params.numIsocontours; i++) {
					//Isocontour::gradientStepping(isocontour, RenderingUtils::getInstance()->switchScalarField2D(m_params.currentScalarVisualization, m_pGridData), m_pGridData, (i + 1)*m_params.isoStepVertical + m_params.initialIsoValue, m_params.isoStepHorizontal);
					//vector<Vector2> *pIsocontourPoints, const Array2D<Scalar> &scalarField, GridData2D *pGridData, Scalar isoValue);
					LevelSets::Isocontour::marchingSquares(&isocontour, RenderingUtils::getInstance()->switchScalarField2D(m_params.currentScalarVisualization, m_pGridData), m_pGridData, (i + 1)*m_params.isoStepVertical + m_params.initialIsoValue);
					m_isoLines.push_back(isocontour);
					isocontour.clear();
				}
			}
			
		}

		#pragma endregion AuxiliaryFunctions
	}
}