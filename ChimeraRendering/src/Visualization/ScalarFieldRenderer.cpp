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


#include "Visualization/ScalarFieldRenderer.h"
#include "RenderingUtils.h"

namespace Chimera {
	namespace Rendering {

		template<>
		ScalarFieldRenderer<Vector2>::ScalarFieldRenderer(StructuredGrid<Vector2> *pGrid): m_isoContourRenderer(pGrid->getGridData2D()) {
			m_pFineGridScalarField2D = NULL;
			m_pFineGridStreamfunction3D = NULL;
			m_fineGridDx = 0.0f;
			m_colorScheme = viridis;
			m_initialIsoValue = 0.01f;
			m_isoStepVertical = 1.0f;
			m_isoStepHorizontal = 0.04;
			m_drawIsoPoints = true;
			m_updateScalarMinMax = true;
			m_triangulateQuads = true;
			m_drawFineGridCells = false;
			m_pCutCells = nullptr;
			m_pCutCellsScalarField = nullptr;

			GridData2D *pGridData2D = pGrid->getGridData2D();
			m_gridDimensions = pGridData2D->getDimensions();
			m_pGrid = pGrid;

			m_totalGridVertices = m_gridDimensions.x*m_gridDimensions.y;

			/** Initialization */
			initializeVBOs();
			initializeShaders();

			if (m_triangulateQuads) {
				int numOfQuadsCentroids = (m_gridDimensions.x - 1)*(m_gridDimensions.y - 1);
				m_pTriangulatedScalarField = new Scalar[m_gridDimensions.x*m_gridDimensions.y + numOfQuadsCentroids];
			}
		}

		template<>
		ScalarFieldRenderer<Vector3>::ScalarFieldRenderer(StructuredGrid<Vector3> *pGrid) : m_isoContourRenderer(pGrid->getGridData2D()) {
			m_pFineGridScalarField2D = NULL;
			m_pFineGridStreamfunction3D = NULL;
			m_fineGridDx = 0.0f;
			m_colorScheme = viridis;
			m_updateScalarMinMax = true;
			m_pCutCells = nullptr;
			m_pCutCellsScalarField = nullptr;
			m_triangulateQuads = false;

			GridData3D *pGridData3D = pGrid->getGridData3D();
			m_gridDimensions = pGridData3D->getDimensions();
			m_pGrid = pGrid;
			
			m_totalGridVertices = m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z;

			/** Initialization */
			initializeVBOs();
			initializeShaders();
		}

		/************************************************************************/
		/* Initialization                                                       */
		/************************************************************************/
		template <class VectorT>
		unsigned int ScalarFieldRenderer<VectorT>::initializeGridCentroidsVBO() {
			void *pGridCentroidsPtr = NULL;

			m_pGridCentroidsVBO = new GLuint();
			glGenBuffers(1, m_pGridCentroidsVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGridCentroidsVBO);
			unsigned int sizeCentroid = m_totalGridVertices*sizeof(VectorT);
			if (m_gridDimensions.z == 0) {
				if (m_triangulateQuads) {
					//Allocating extra space for centroid centroids which are going to be used to triangulate quads
					int extraCentroidsSize = (m_pGrid->getGridData2D()->getDimensions().x - 1)*(m_pGrid->getGridData2D()->getDimensions().y - 1);
					sizeCentroid = (m_totalGridVertices + extraCentroidsSize)*sizeof(VectorT);
				}
			}

			if(m_gridDimensions.z == 0) {
				pGridCentroidsPtr = reinterpret_cast<void *>(m_pGrid->getGridData2D()->getGridCentersArray().getRawDataPointer());
			} else {
				pGridCentroidsPtr = reinterpret_cast<void *>(m_pGrid->getGridData3D()->getGridCentersArray().getRawDataPointer());
			}

			glBufferData(GL_ARRAY_BUFFER, sizeCentroid, 0, GL_STATIC_DRAW);
			glBufferSubData(GL_ARRAY_BUFFER, 0, m_totalGridVertices*sizeof(VectorT), pGridCentroidsPtr);

			if (m_triangulateQuads) {
				if (m_gridDimensions.z == 0) {
					vector<VectorT> triangleCentroids((m_pGrid->getGridData2D()->getDimensions().x - 1)*(m_pGrid->getGridData2D()->getDimensions().y - 1));
					int dimX = m_pGrid->getGridData2D()->getDimensions().x - 1;
					int dimY = m_pGrid->getGridData2D()->getDimensions().y - 1;
					const Array2D<Vector2> &centersArray = m_pGrid->getGridData2D()->getGridCentersArray();
					for (int i = 0; i < dimX; i++) {
						for (int j = 0; j < dimY; j++) {
							triangleCentroids[j*dimX + i] = centersArray(i, j) + centersArray(i + 1, j) + centersArray(i, j + 1) + centersArray(i + 1, j + 1);
							triangleCentroids[j*dimX + i] *= 0.25;
						}
					}
					int offset = m_totalGridVertices*sizeof(VectorT);
					int triangleCentroidsSize = (m_pGrid->getGridData2D()->getDimensions().x - 1)*(m_pGrid->getGridData2D()->getDimensions().y - 1)*sizeof(VectorT);
					glBufferSubData(GL_ARRAY_BUFFER, offset, triangleCentroidsSize, (void *)&triangleCentroids[0]);
					//delete triangleCentroids;
				}
			}
			
			return sizeCentroid;
		}

		template <class VectorT>
		unsigned int ScalarFieldRenderer<VectorT>::initializeScalarFieldVBO() {
			m_pScalarFieldVBO = new GLuint();
			glGenBuffers(1, m_pScalarFieldVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarFieldVBO);
			unsigned int size = m_totalGridVertices*sizeof(Scalar);
			if (m_triangulateQuads && m_gridDimensions.z == 0) {
				int extraCentroidsSize = (m_gridDimensions.x - 1)*(m_gridDimensions.y - 1);
				size = (m_totalGridVertices + extraCentroidsSize)*sizeof(Scalar);
			}
			glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			m_pScalarColorsVBO = new GLuint();
			glGenBuffers(1, m_pScalarColorsVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarColorsVBO);
			unsigned int sizeColors = m_totalGridVertices*sizeof(Vector3);
			if (m_triangulateQuads == m_gridDimensions.z == 0) {
				int extraCentroidsSize = (m_gridDimensions.x - 1)*(m_gridDimensions.y - 1);
				sizeColors = (m_totalGridVertices + extraCentroidsSize)*sizeof(Vector3);
			}
			glBufferData(GL_ARRAY_BUFFER, sizeColors, 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			return size + sizeColors;
		}

		template <>
		unsigned int ScalarFieldRenderer<Vector2>::initializeScalarIndexVBO() {
			m_pScalarIndexXYVBO = new GLuint();
			glGenBuffers(1, m_pScalarIndexXYVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarIndexXYVBO);

			unsigned int size, sizeX;

			if(m_pGrid->isPeriodic()) {
				size = (m_gridDimensions.x)*(m_gridDimensions.y - 1)*4;
				sizeX = m_gridDimensions.x;
			} else if(m_triangulateQuads) {
				size = (m_gridDimensions.x - 1)*(m_gridDimensions.y - 1) * 4 * 3;
				sizeX = m_gridDimensions.x - 1;
			} else {
				size = (m_gridDimensions.x - 1)*(m_gridDimensions.y - 1)*4;
				sizeX = m_gridDimensions.x - 1;
			}

			int *pIndexes = new int[size];
			for(unsigned int i = 0; i < size; i++) {
				pIndexes[i] = 0;
			}
			int currIndex = 0;
			/* Triangulate quad scheme
				(i, j + 1) (i + 1, j + 1)
				* ---- *
				| \  / |
				|  *   |
				| /  \ |
				* -----*
				(i, j) (i + 1, j)
				Center point = (m_gridDimensions.x - 1)*(m_gridDimensions.y - 1) + cellIndex(i, j)
				BottomTriangle = (i, j), (i + 1, j), centroid
				LeftTriangle = (i + 1, j), (i + 1, j + 1), centroid
				TopTriangle  = (i + 1, j + 1), (i, j + 1), centroid
				RightTriangle = (i, j + 1), (i, j), centroid
			*/
			if (m_triangulateQuads) {
				int initialOffset = m_gridDimensions.x*m_gridDimensions.y;
				for (int i = 0; i < m_gridDimensions.x - 1; i++) {
					for (int j = 0; j < m_gridDimensions.y - 1; j++) {
						//Bottom triangle
						currIndex = (j*sizeX + i) * 4 * 3;
						pIndexes[currIndex] = getRegularGridIndex(i, j);
						pIndexes[currIndex + 1] = getRegularGridIndex(i + 1, j);
						pIndexes[currIndex + 2] = initialOffset + j*sizeX + i; //Centroid index
						//Left triangle
						pIndexes[currIndex + 3] = getRegularGridIndex(i + 1, j);
						pIndexes[currIndex + 4] = getRegularGridIndex(i + 1, j + 1);
						pIndexes[currIndex + 5] = initialOffset + j*sizeX + i; //Centroid index
						//Top triangle
						pIndexes[currIndex + 6] = getRegularGridIndex(i + 1, j + 1);
						pIndexes[currIndex + 7] = getRegularGridIndex(i, j + 1);
						pIndexes[currIndex + 8] = initialOffset + j*sizeX + i; //Centroid index
						//Right triangle
						pIndexes[currIndex + 9] = getRegularGridIndex(i, j + 1);
						pIndexes[currIndex + 10] = getRegularGridIndex(i, j);
						pIndexes[currIndex + 11] = initialOffset + j*sizeX + i; //Centroid index

					}
				}
			}
			else {
				for (int i = 0; i < m_gridDimensions.x - 1; i++) {
					for (int j = 0; j < m_gridDimensions.y - 1; j++) {
						currIndex = (j*sizeX + i) * 4;
						pIndexes[currIndex] = j*(m_gridDimensions.x) + i;
						pIndexes[currIndex + 1] = j*(m_gridDimensions.x) + i + 1;
						pIndexes[currIndex + 2] = (j + 1)*(m_gridDimensions.x) + i + 1;
						pIndexes[currIndex + 3] = (j + 1)*(m_gridDimensions.x) + i;
					}
				}
			}
			

			if(m_pGrid->isPeriodic()) {
				for(int j = 0; j < m_gridDimensions.y - 1; j++) {
					currIndex = (j*sizeX + m_gridDimensions.x - 1)*4;
					pIndexes[currIndex]		= j*(m_gridDimensions.x) + m_gridDimensions.x - 1;
					pIndexes[currIndex + 1] = j*(m_gridDimensions.x) + 0;
					pIndexes[currIndex + 2] = (j + 1)*(m_gridDimensions.x) + 0;
					pIndexes[currIndex + 3] = (j + 1)*(m_gridDimensions.x) + m_gridDimensions.x - 1;
				}
			}

			glBufferData(GL_ARRAY_BUFFER, size*sizeof(int), pIndexes, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			delete[] pIndexes;
			return size*sizeof(int);
		}

		template <>
		unsigned int ScalarFieldRenderer<Vector3>::initializeScalarIndexVBO() {
			m_pScalarIndexXYVBO = new GLuint();
			glGenBuffers(1, m_pScalarIndexXYVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarIndexXYVBO);

			GridData3D *pGridData3D = m_pGrid->getGridData3D();
			unsigned int totalSize = 0;

			/************************************************************************/
			/* XY indices initialization                                            */
			/************************************************************************/
			unsigned int size = (pGridData3D->getDimensions().x - 1)*(pGridData3D->getDimensions().y - 1)*(pGridData3D->getDimensions().z)*4;
			int *pIndexes = new int[size];
			int currIndex = 0;
			const Array3D<Vector3> &velocityArray = pGridData3D->getVelocityArray();
			for(int i = 0; i < pGridData3D->getDimensions().x - 1; i++) {
				for(int j = 0; j < pGridData3D->getDimensions().y - 1; j++) {
					for(int k = 0; k <pGridData3D->getDimensions().z; k++) {
						currIndex				= (k*(pGridData3D->getDimensions().x - 1)*(pGridData3D->getDimensions().y - 1) + j*(pGridData3D->getDimensions().x - 1) + i)*4;
						pIndexes[currIndex]		= velocityArray.getLinearIndex(i, j, k);
						pIndexes[currIndex + 1] = velocityArray.getLinearIndex(i + 1, j, k); 
						pIndexes[currIndex + 2] = velocityArray.getLinearIndex(i + 1, j + 1, k); 
						pIndexes[currIndex + 3] = velocityArray.getLinearIndex(i, j + 1, k); 
					}	
				}
			}

			
			glBufferData(GL_ARRAY_BUFFER, size*sizeof(int), pIndexes, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			delete[] pIndexes;
			totalSize += size;

			/************************************************************************/
			/* YZ indices initialization                                            */
			/************************************************************************/
			m_pScalarIndexYZVBO = new GLuint();
			glGenBuffers(1, m_pScalarIndexYZVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarIndexYZVBO);
			size = (pGridData3D->getDimensions().x)*(pGridData3D->getDimensions().y - 1)*(pGridData3D->getDimensions().z - 1) * 4;
			pIndexes = new int[size];
			currIndex = 0;

			for (int i = 0; i < pGridData3D->getDimensions().x; i++) {
				for (int j = 0; j < pGridData3D->getDimensions().y - 1; j++) {
					for (int k = 0; k < pGridData3D->getDimensions().z - 1; k++) {
						//currIndex = (i*(pGridData3D->getDimensions().z - 1)*(pGridData3D->getDimensions().y - 1) + j*(pGridData3D->getDimensions().z - 1) + k) * 4;
						pIndexes[currIndex++] = velocityArray.getLinearIndex(i, j, k);
						pIndexes[currIndex++] = velocityArray.getLinearIndex(i, j + 1, k);
						pIndexes[currIndex++] = velocityArray.getLinearIndex(i, j + 1, k + 1);
						pIndexes[currIndex++] = velocityArray.getLinearIndex(i, j, k + 1);
					}
				}
			}

			glBufferData(GL_ARRAY_BUFFER, size*sizeof(int), pIndexes, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			delete[] pIndexes;
			totalSize += size;

			/************************************************************************/
			/* XZ indices initialization                                            */
			/************************************************************************/
			m_pScalarIndexXZVBO = new GLuint();
			glGenBuffers(1, m_pScalarIndexXZVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarIndexXZVBO);
			size = (pGridData3D->getDimensions().x - 1)*(pGridData3D->getDimensions().y)*(pGridData3D->getDimensions().z - 1) * 4;
			pIndexes = new int[size];
			currIndex = 0;
			for (int j = 0; j < pGridData3D->getDimensions().y; j++) {
				for (int i = 0; i < pGridData3D->getDimensions().x - 1; i++) {
					for (int k = 0; k < pGridData3D->getDimensions().z - 1; k++) {
						//currIndex = (k*(pGridData3D->getDimensions().x - 1)*(pGridData3D->getDimensions().y - 1) + j*(pGridData3D->getDimensions().x - 1) + i) * 4;
						pIndexes[currIndex++] = velocityArray.getLinearIndex(i, j, k);
						pIndexes[currIndex++] = velocityArray.getLinearIndex(i + 1, j, k);
						pIndexes[currIndex++] = velocityArray.getLinearIndex(i + 1, j, k + 1);
						pIndexes[currIndex++] = velocityArray.getLinearIndex(i, j, k + 1);
					}
				}
			}

			glBufferData(GL_ARRAY_BUFFER, size*sizeof(int), pIndexes, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			delete[] pIndexes;
			totalSize += size;

			return totalSize*sizeof(int);
		}

		template <>
		void ScalarFieldRenderer<Vector2>::initializeFineGridScalarVBOs() {
			/**Initialize streamfunction vertices */
			Array2D<Vector2> streamfunctionVertices(m_pFineGridScalarField2D->getDimensions());
			void *pStreamfunctionVertices = NULL;

			m_pFineGridVerticesVBO = new GLuint();
			glGenBuffers(1, m_pFineGridVerticesVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pFineGridVerticesVBO);
			int totalNumberOfVertices = m_pFineGridScalarField2D->getDimensions().x*m_pFineGridScalarField2D->getDimensions().y;
			unsigned int sizeVertices = totalNumberOfVertices*sizeof(Vector2);

			for (int i = 0; i < m_pFineGridScalarField2D->getDimensions().x; i++) {
				for (int j = 0; j < m_pFineGridScalarField2D->getDimensions().y; j++) {
					streamfunctionVertices(i, j) = Vector2(i*m_fineGridDx, j*m_fineGridDx);
				}
			}

			glBufferData(GL_ARRAY_BUFFER, sizeVertices, 0, GL_STATIC_DRAW);
			glBufferSubData(GL_ARRAY_BUFFER, 0, totalNumberOfVertices*sizeof(Vector2), (void *) streamfunctionVertices.getRawDataPointer());

			/* Values */
			m_pFineGridValuesVBO = new GLuint();
			glGenBuffers(1, m_pFineGridValuesVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pFineGridValuesVBO);
			unsigned int size = totalNumberOfVertices*sizeof(Scalar);

			glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			/* Collors */
			m_pFineGridColorsVBO = new GLuint();
			glGenBuffers(1, m_pFineGridColorsVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pFineGridColorsVBO);
			unsigned int sizeColors = totalNumberOfVertices*sizeof(Vector3);
			glBufferData(GL_ARRAY_BUFFER, sizeColors, 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		
			/* Indices */
			m_pFineGridIndexVBO = new GLuint();
			glGenBuffers(1, m_pFineGridIndexVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pFineGridIndexVBO);

			unsigned int sizeIndices = (m_pFineGridScalarField2D->getDimensions().x-1)*(m_pFineGridScalarField2D->getDimensions().y-1)*4;
			int *pIndexes = new int[sizeIndices];

			for (int i = 0; i < m_pFineGridScalarField2D->getDimensions().x - 1; i++) {
				for (int j = 0; j < m_pFineGridScalarField2D->getDimensions().y - 1; j++) {
					int currIndex = (j*(m_pFineGridScalarField2D->getDimensions().x - 1) + i) * 4;
					pIndexes[currIndex] = j*(m_pFineGridScalarField2D->getDimensions().x) + i;
					pIndexes[currIndex + 1] = j*(m_pFineGridScalarField2D->getDimensions().x) + i + 1;
					pIndexes[currIndex + 2] = (j + 1)*(m_pFineGridScalarField2D->getDimensions().x) + i + 1;
					pIndexes[currIndex + 3] = (j + 1)*(m_pFineGridScalarField2D->getDimensions().x) + i;
				}
			}

			glBufferData(GL_ARRAY_BUFFER, sizeIndices*sizeof(int), pIndexes, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			delete[] pIndexes;
		}

		template <>
		void ScalarFieldRenderer<Vector3>::initializeFineGridScalarVBOs() {

		}
		/************************************************************************/
		/* Shaders initializtion	                                            */
		/************************************************************************/
		template <class VectorT>
		void ScalarFieldRenderer<VectorT>::initializeShaders() {
			/** Viridis color shader */
			{
				GLchar const * Strings[] = { "rColor", "gColor", "bColor" };
				m_pViridisColorShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER,
																						"Shaders/2D/ScalarColor - viridis.glsl",
																						3,
																						Strings,
																						GL_INTERLEAVED_ATTRIBS);

				m_virMinScalarLoc = glGetUniformLocation(m_pViridisColorShader->getProgramID(), "minPressure");
				m_virMaxScalarLoc = glGetUniformLocation(m_pViridisColorShader->getProgramID(), "maxPressure");
				m_virAvgScalarLoc = glGetUniformLocation(m_pViridisColorShader->getProgramID(), "avgPressure");
			}
			
			/** Jet color shader */
			{
				GLchar const * Strings[] = {"rColor", "gColor", "bColor"}; 
				m_pJetColorShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER, 
					"Shaders/2D/ScalarColor - wavelength.glsl",
					3,
					Strings,
					GL_INTERLEAVED_ATTRIBS);

				m_jetMinScalarLoc = glGetUniformLocation(m_pJetColorShader->getProgramID(), "minPressure");
				m_jetMaxScalarLoc = glGetUniformLocation(m_pJetColorShader->getProgramID(), "maxPressure");
				m_jetAvgScalarLoc = glGetUniformLocation(m_pJetColorShader->getProgramID(), "avgPressure");
			}

			/** Grayscale color shader */
			{
				GLchar const * Strings[] = {"rColor", "gColor", "bColor"}; 
				m_pGrayScaleColorShader = ResourceManager::getInstance()->loadGLSLShader(GL_VERTEX_SHADER, 
					"Shaders/2D/ScalarColor - grayscale.glsl",
					3,
					Strings,
					GL_INTERLEAVED_ATTRIBS);

				m_grayMinScalarLoc = glGetUniformLocation(m_pGrayScaleColorShader->getProgramID(), "minScalar");
				m_grayMaxScalarLoc = glGetUniformLocation(m_pGrayScaleColorShader->getProgramID(), "maxScalar");
			}

		}

		
		/************************************************************************/
		/* Update                                                               */
		/************************************************************************/
		template <>
		void ScalarFieldRenderer<Vector2>::updateValueColor(const BaseWindow::scalarVisualization_t &visualizationType) {
			void *pValuePtr;
			if (visualizationType == BaseWindow::scalarVisualization_t::drawFineGridScalars) {
				pValuePtr = m_pFineGridScalarField2D->getRawDataPointer();
				glBindBuffer(GL_ARRAY_BUFFER, *m_pFineGridValuesVBO);
			} else {
				pValuePtr = RenderingUtils::getInstance()->switchScalarField2D(visualizationType, m_pGrid->getGridData2D()).getRawDataPointer();
				glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarFieldVBO);
			}
			int totalTriangulatedVertices = m_totalGridVertices + (m_gridDimensions.x - 1)*(m_gridDimensions.y - 1);
			/*Updating scalar field values inside scalarFieldVBO*/
			if (visualizationType == BaseWindow::scalarVisualization_t::drawFineGridScalars) {
				glBufferData(GL_ARRAY_BUFFER, m_pFineGridScalarField2D->getDimensions().x*m_pFineGridScalarField2D->getDimensions().y*sizeof(Scalar), pValuePtr, GL_DYNAMIC_DRAW);
			} else if (m_triangulateQuads) {
				updateTriangulatedScalarField((Scalar*) pValuePtr);
				glBufferData(GL_ARRAY_BUFFER, totalTriangulatedVertices*sizeof(Scalar), m_pTriangulatedScalarField, GL_DYNAMIC_DRAW);
			}
			else {
				glBufferData(GL_ARRAY_BUFFER, m_totalGridVertices*sizeof(Scalar), pValuePtr, GL_DYNAMIC_DRAW);
			}
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			/** Avg and max pressure calculation */
			Scalar avgValue = 0.5*(m_minScalarFieldVal + m_maxScalarFieldVal);
			applyColorShader(m_minScalarFieldVal, m_maxScalarFieldVal, avgValue);

			glEnable(GL_RASTERIZER_DISCARD_NV);
			if (visualizationType == BaseWindow::scalarVisualization_t::drawFineGridScalars) {
				glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, *m_pFineGridColorsVBO);

				glBindBuffer(GL_ARRAY_BUFFER, *m_pFineGridValuesVBO);
			}
			else {
				glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, *m_pScalarColorsVBO);

				glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarFieldVBO);
			}
			
			glVertexAttribPointer(0, 1, GL_FLOAT, false, 0, 0);
			glEnableVertexAttribArray(0);

			glBeginTransformFeedback(GL_POINTS);
			if (visualizationType == BaseWindow::scalarVisualization_t::drawFineGridScalars) {
				glDrawArrays(GL_POINTS, 0, m_pFineGridScalarField2D->getDimensions().x*m_pFineGridScalarField2D->getDimensions().y);
			} else if (m_triangulateQuads) {
				glDrawArrays(GL_POINTS, 0, totalTriangulatedVertices);
			}
			else {
				glDrawArrays(GL_POINTS, 0, m_totalGridVertices);
			}
			glEndTransformFeedback();
			glDisableVertexAttribArray(0);

			glDisable(GL_RASTERIZER_DISCARD_NV);
			removeColorShader();
		}

		template <>
		void ScalarFieldRenderer<Vector3>::updateValueColor(const BaseWindow::scalarVisualization_t &visualizationType) {
			void *pValuePtr;
			if (visualizationType == BaseWindow::scalarVisualization_t::drawFineGridScalars) {
				pValuePtr = m_pFineGridStreamfunction3D->getRawDataPointer();
			} else {
				pValuePtr = RenderingUtils::getInstance()->switchScalarField3D(visualizationType, m_pGrid->getGridData3D()).getRawDataPointer();
			}
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarFieldVBO);
			int totalTriangulatedVertices = m_totalGridVertices + (m_gridDimensions.x - 1)*(m_gridDimensions.y - 1);
			/*Updating scalar field values inside scalarFieldVBO*/
			if (m_triangulateQuads) {
				updateTriangulatedScalarField((Scalar*)pValuePtr);
				glBufferData(GL_ARRAY_BUFFER, totalTriangulatedVertices*sizeof(Scalar), m_pTriangulatedScalarField, GL_DYNAMIC_DRAW);
			}
			else {
				glBufferData(GL_ARRAY_BUFFER, m_totalGridVertices*sizeof(Scalar), pValuePtr, GL_DYNAMIC_DRAW);
			}
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			/** Avg and max pressure calculation */
			Scalar avgValue = 0.5*(m_minScalarFieldVal + m_maxScalarFieldVal);
			applyColorShader(m_minScalarFieldVal, m_maxScalarFieldVal, avgValue);

			glEnable(GL_RASTERIZER_DISCARD_NV);
			glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, *m_pScalarColorsVBO);

			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarFieldVBO);
			glVertexAttribPointer(0, 1, GL_FLOAT, false, 0, 0);
			glEnableVertexAttribArray(0);

			glBeginTransformFeedback(GL_POINTS);
			if (m_triangulateQuads) {
				glDrawArrays(GL_POINTS, 0, totalTriangulatedVertices);
			}
			else {
				glDrawArrays(GL_POINTS, 0, m_totalGridVertices);
			}
			glEndTransformFeedback();
			glDisableVertexAttribArray(0);

			glDisable(GL_RASTERIZER_DISCARD_NV);
			removeColorShader();
		}

		/************************************************************************/
		/* Private functionalities                                              */
		/************************************************************************/
		template <>
		void ScalarFieldRenderer<Vector2>::findMinMax(const Array<Scalar> &scalarField, Scalar &minValue, Scalar &maxValue) const {
			Array2D<Scalar> *pScalarField2D = (Array2D<Scalar> *) &scalarField;
			maxValue = -1e10;
			minValue = 1e10;

			for (int i = 0; i < scalarField.getDimensions().x; i++) {
				for (int j = 0; j < scalarField.getDimensions().y; j++) {
					if((*pScalarField2D)(i, j) > maxValue) 
						maxValue = (*pScalarField2D)(i, j);

					if((*pScalarField2D)(i, j) < minValue)
						minValue = (*pScalarField2D)(i, j);
				}
			}

			if (m_pCutCells && m_pCutCellsScalarField) {
				for (int i = 0; i < m_pCutCellsScalarField->size(); i++) {
					if ((*m_pCutCellsScalarField)[i] > maxValue)
						maxValue = (*m_pCutCellsScalarField)[i];

					if ((*m_pCutCellsScalarField)[i] < minValue)
						minValue = (*m_pCutCellsScalarField)[i];
				}
			}
		}

		template <>
		void ScalarFieldRenderer<Vector3>::findMinMax(const Array<Scalar> &scalarField, Scalar &minValue, Scalar &maxValue) const {
			Array3D<Scalar> *pScalarField3D = (Array3D<Scalar> *) &scalarField;
			maxValue = -1e10;
			minValue = 1e10;

			for(int i = 0; i < m_gridDimensions.x; i++) {
				for(int j = 0; j < m_gridDimensions.y; j++) {
					for(int k = 0; k < m_gridDimensions.z; k++) {
						if((*pScalarField3D)(i, j, k) > maxValue) 
							maxValue = (*pScalarField3D)(i, j, k);

						if((*pScalarField3D)(i, j, k) < minValue)
							minValue = (*pScalarField3D)(i, j, k);
					}
				}
			}

			if (m_pCutCells && m_pCutCellsScalarField) {
				for (int i = 0; i < m_pCutCellsScalarField->size(); i++) {
					if ((*m_pCutCellsScalarField)[i] > maxValue)
						maxValue = (*m_pCutCellsScalarField)[i];

					if ((*m_pCutCellsScalarField)[i] < minValue)
						minValue = (*m_pCutCellsScalarField)[i];
				}
			}
		}

		template <class VectorT>
		void ScalarFieldRenderer<VectorT>::updateTriangulatedScalarField(Scalar *pScalarField) {
			for (int i = 0; i < m_gridDimensions.x; i++) {
				for (int j = 0; j < m_gridDimensions.y; j++) {
					m_pTriangulatedScalarField[getRegularGridIndex(i, j)] = pScalarField[getRegularGridIndex(i, j)];
				}
			}
			int initialIndex = m_gridDimensions.x*m_gridDimensions.y;
			for (int i = 0; i < m_gridDimensions.x - 1; i++) {
				for (int j = 0; j < m_gridDimensions.y - 1; j++) {
					Scalar avgValue = pScalarField[getRegularGridIndex(i, j)] + pScalarField[getRegularGridIndex(i + 1, j)] 
									+ pScalarField[getRegularGridIndex(i + 1, j + 1)] + pScalarField[getRegularGridIndex(i, j + 1)];
					avgValue *= 0.25;

					int currIndex = initialIndex + (m_gridDimensions.x - 1)*j + i;
					m_pTriangulatedScalarField[currIndex] = avgValue;
				}
			}
		}

		template<>
		Vector3 ScalarFieldRenderer<Vector2>::viridisColorMap(Scalar scalarValue) {
			const static float rColorVec[256] = { 0.26700401, 0.26851048, 0.26994384, 0.27130489, 0.27259384, 0.27380934, 0.27495242, 0.27602238, 0.2770184 , 0.27794143, 0.27879067, 0.2795655 , 0.28026658, 0.28089358, 0.28144581, 0.28192358, 0.28232739, 0.28265633, 0.28291049, 0.28309095, 0.28319704, 0.28322882, 0.28318684, 0.283072  , 0.28288389, 0.28262297, 0.28229037, 0.28188676, 0.28141228, 0.28086773, 0.28025468, 0.27957399, 0.27882618, 0.27801236, 0.27713437, 0.27619376, 0.27519116, 0.27412802, 0.27300596, 0.27182812, 0.27059473, 0.26930756, 0.26796846, 0.26657984, 0.2651445 , 0.2636632 , 0.26213801, 0.26057103, 0.25896451, 0.25732244, 0.25564519, 0.25393498, 0.25219404, 0.25042462, 0.24862899, 0.2468114 , 0.24497208, 0.24311324, 0.24123708, 0.23934575, 0.23744138, 0.23552606, 0.23360277, 0.2316735 , 0.22973926, 0.22780192, 0.2258633 , 0.22392515, 0.22198915, 0.22005691, 0.21812995, 0.21620971, 0.21429757, 0.21239477, 0.2105031 , 0.20862342, 0.20675628, 0.20490257, 0.20306309, 0.20123854, 0.1994295 , 0.1976365 , 0.19585993, 0.19410009, 0.19235719, 0.19063135, 0.18892259, 0.18723083, 0.18555593, 0.18389763, 0.18225561, 0.18062949, 0.17901879, 0.17742298, 0.17584148, 0.17427363, 0.17271876, 0.17117615, 0.16964573, 0.16812641, 0.1666171 , 0.16511703, 0.16362543, 0.16214155, 0.16066467, 0.15919413, 0.15772933, 0.15626973, 0.15481488, 0.15336445, 0.1519182 , 0.15047605, 0.14903918, 0.14760731, 0.14618026, 0.14475863, 0.14334327, 0.14193527, 0.14053599, 0.13914708, 0.13777048, 0.1364085 , 0.13506561, 0.13374299, 0.13244401, 0.13117249, 0.1299327 , 0.12872938, 0.12756771, 0.12645338, 0.12539383, 0.12439474, 0.12346281, 0.12260562, 0.12183122, 0.12114807, 0.12056501, 0.12009154, 0.11973756, 0.11951163, 0.11942341, 0.11948255, 0.11969858, 0.12008079, 0.12063824, 0.12137972, 0.12231244, 0.12344358, 0.12477953, 0.12632581, 0.12808703, 0.13006688, 0.13226797, 0.13469183, 0.13733921, 0.14020991, 0.14330291, 0.1466164 , 0.15014782, 0.15389405, 0.15785146, 0.16201598, 0.1663832 , 0.1709484 , 0.17570671, 0.18065314, 0.18578266, 0.19109018, 0.19657063, 0.20221902, 0.20803045, 0.21400015, 0.22012381, 0.2263969 , 0.23281498, 0.2393739 , 0.24606968, 0.25289851, 0.25985676, 0.26694127, 0.27414922, 0.28147681, 0.28892102, 0.29647899, 0.30414796, 0.31192534, 0.3198086 , 0.3277958 , 0.33588539, 0.34407411, 0.35235985, 0.36074053, 0.3692142 , 0.37777892, 0.38643282, 0.39517408, 0.40400101, 0.4129135 , 0.42190813, 0.43098317, 0.44013691, 0.44936763, 0.45867362, 0.46805314, 0.47750446, 0.4870258 , 0.49661536, 0.5062713 , 0.51599182, 0.52577622, 0.5356211 , 0.5455244 , 0.55548397, 0.5654976 , 0.57556297, 0.58567772, 0.59583934, 0.60604528, 0.61629283, 0.62657923, 0.63690157, 0.64725685, 0.65764197, 0.66805369, 0.67848868, 0.68894351, 0.69941463, 0.70989842, 0.72039115, 0.73088902, 0.74138803, 0.75188414, 0.76237342, 0.77285183, 0.78331535, 0.79375994, 0.80418159, 0.81457634, 0.82494028, 0.83526959, 0.84556056, 0.8558096 , 0.86601325, 0.87616824, 0.88627146, 0.89632002, 0.90631121, 0.91624212, 0.92610579, 0.93590444, 0.94563626, 0.95529972, 0.96489353, 0.97441665, 0.98386829, 0.99324789 };
			const static float gColorVec[256] = { 0.00487433, 0.00960483, 0.01462494, 0.01994186, 0.02556309, 0.03149748, 0.03775181, 0.04416723, 0.05034437, 0.05632444, 0.06214536, 0.06783587, 0.07341724, 0.07890703, 0.0843197 , 0.08966622, 0.09495545, 0.10019576, 0.10539345, 0.11055307, 0.11567966, 0.12077701, 0.12584799, 0.13089477, 0.13592005, 0.14092556, 0.14591233, 0.15088147, 0.15583425, 0.16077132, 0.16569272, 0.17059884, 0.1754902 , 0.18036684, 0.18522836, 0.19007447, 0.1949054 , 0.19972086, 0.20452049, 0.20930306, 0.21406899, 0.21881782, 0.22354911, 0.2282621 , 0.23295593, 0.23763078, 0.24228619, 0.2469217 , 0.25153685, 0.2561304 , 0.26070284, 0.26525384, 0.26978306, 0.27429024, 0.27877509, 0.28323662, 0.28767547, 0.29209154, 0.29648471, 0.30085494, 0.30520222, 0.30952657, 0.31382773, 0.3181058 , 0.32236127, 0.32659432, 0.33080515, 0.334994  , 0.33916114, 0.34330688, 0.34743154, 0.35153548, 0.35561907, 0.35968273, 0.36372671, 0.36775151, 0.37175775, 0.37574589, 0.37971644, 0.38366989, 0.38760678, 0.39152762, 0.39543297, 0.39932336, 0.40319934, 0.40706148, 0.41091033, 0.41474645, 0.4185704 , 0.42238275, 0.42618405, 0.42997486, 0.43375572, 0.4375272 , 0.44128981, 0.4450441 , 0.4487906 , 0.4525298 , 0.45626209, 0.45998802, 0.46370813, 0.4674229 , 0.47113278, 0.47483821, 0.47853961, 0.4822374 , 0.48593197, 0.4896237 , 0.49331293, 0.49700003, 0.50068529, 0.50436904, 0.50805136, 0.51173263, 0.51541316, 0.51909319, 0.52277292, 0.52645254, 0.53013219, 0.53381201, 0.53749213, 0.54117264, 0.54485335, 0.54853458, 0.55221637, 0.55589872, 0.55958162, 0.56326503, 0.56694891, 0.57063316, 0.57431754, 0.57800205, 0.58168661, 0.58537105, 0.58905521, 0.59273889, 0.59642187, 0.60010387, 0.60378459, 0.60746388, 0.61114146, 0.61481702, 0.61849025, 0.62216081, 0.62582833, 0.62949242, 0.63315277, 0.63680899, 0.64046069, 0.64410744, 0.64774881, 0.65138436, 0.65501363, 0.65863619, 0.66225157, 0.66585927, 0.66945881, 0.67304968, 0.67663139, 0.68020343, 0.68376525, 0.68731632, 0.69085611, 0.69438405, 0.6978996 , 0.70140222, 0.70489133, 0.70836635, 0.71182668, 0.71527175, 0.71870095, 0.72211371, 0.72550945, 0.72888753, 0.73224735, 0.73558828, 0.73890972, 0.74221104, 0.74549162, 0.74875084, 0.75198807, 0.75520266, 0.75839399, 0.76156142, 0.76470433, 0.76782207, 0.77091403, 0.77397953, 0.7770179 , 0.78002855, 0.78301086, 0.78596419, 0.78888793, 0.79178146, 0.79464415, 0.79747541, 0.80027461, 0.80304099, 0.80577412, 0.80847343, 0.81113836, 0.81376835, 0.81636288, 0.81892143, 0.82144351, 0.82392862, 0.82637633, 0.82878621, 0.83115784, 0.83349064, 0.83578452, 0.83803918, 0.84025437, 0.8424299 , 0.84456561, 0.84666139, 0.84871722, 0.8507331 , 0.85270912, 0.85464543, 0.85654226, 0.85839991, 0.86021878, 0.86199932, 0.86374211, 0.86544779, 0.86711711, 0.86875092, 0.87035015, 0.87191584, 0.87344918, 0.87495143, 0.87642392, 0.87786808, 0.87928545, 0.88067763, 0.88204632, 0.88339329, 0.88472036, 0.88602943, 0.88732243, 0.88860134, 0.88986815, 0.89112487, 0.89237353, 0.89361614, 0.89485467, 0.89609127, 0.89732977, 0.8985704 , 0.899815  , 0.90106534, 0.90232311, 0.90358991, 0.90486726, 0.90615657 };
			const static float bColorVec[256] = { 0.32941519, 0.33542652, 0.34137895, 0.34726862, 0.35309303, 0.35885256, 0.36454323, 0.37016418, 0.37571452, 0.38119074, 0.38659204, 0.39191723, 0.39716349, 0.40232944, 0.40741404, 0.41241521, 0.41733086, 0.42216032, 0.42690202, 0.43155375, 0.43611482, 0.44058404, 0.44496   , 0.44924127, 0.45342734, 0.45751726, 0.46150995, 0.46540474, 0.46920128, 0.47289909, 0.47649762, 0.47999675, 0.48339654, 0.48669702, 0.48989831, 0.49300074, 0.49600488, 0.49891131, 0.50172076, 0.50443413, 0.50705243, 0.50957678, 0.5120084 , 0.5143487 , 0.5165993 , 0.51876163, 0.52083736, 0.52282822, 0.52473609, 0.52656332, 0.52831152, 0.52998273, 0.53157905, 0.53310261, 0.53455561, 0.53594093, 0.53726018, 0.53851561, 0.53970946, 0.54084398, 0.5419214 , 0.54294396, 0.54391424, 0.54483444, 0.54570633, 0.546532  , 0.54731353, 0.54805291, 0.54875211, 0.54941304, 0.55003755, 0.55062743, 0.5511844 , 0.55171011, 0.55220646, 0.55267486, 0.55311653, 0.55353282, 0.55392505, 0.55429441, 0.55464205, 0.55496905, 0.55527637, 0.55556494, 0.55583559, 0.55608907, 0.55632606, 0.55654717, 0.55675292, 0.55694377, 0.5571201 , 0.55728221, 0.55743035, 0.55756466, 0.55768526, 0.55779216, 0.55788532, 0.55796464, 0.55803034, 0.55808199, 0.55811913, 0.55814141, 0.55814842, 0.55813967, 0.55811466, 0.5580728 , 0.55801347, 0.557936  , 0.55783967, 0.55772371, 0.55758733, 0.55742968, 0.5572505 , 0.55704861, 0.55682271, 0.55657181, 0.55629491, 0.55599097, 0.55565893, 0.55529773, 0.55490625, 0.55448339, 0.55402906, 0.55354108, 0.55301828, 0.55245948, 0.55186354, 0.55122927, 0.55055551, 0.5498411 , 0.54908564, 0.5482874 , 0.54744498, 0.54655722, 0.54562298, 0.54464114, 0.54361058, 0.54253043, 0.54139999, 0.54021751, 0.53898192, 0.53769219, 0.53634733, 0.53494633, 0.53348834, 0.53197275, 0.53039808, 0.52876343, 0.52706792, 0.52531069, 0.52349092, 0.52160791, 0.51966086, 0.5176488 , 0.51557101, 0.5134268 , 0.51121549, 0.50893644, 0.5065889 , 0.50417217, 0.50168574, 0.49912906, 0.49650163, 0.49380294, 0.49103252, 0.48818938, 0.48527326, 0.48228395, 0.47922108, 0.47608431, 0.4728733 , 0.46958774, 0.46622638, 0.46278934, 0.45927675, 0.45568838, 0.45202405, 0.44828355, 0.44446673, 0.44057284, 0.4366009 , 0.43255207, 0.42842626, 0.42422341, 0.41994346, 0.41558638, 0.41115215, 0.40664011, 0.40204917, 0.39738103, 0.39263579, 0.38781353, 0.38291438, 0.3779385 , 0.37288606, 0.36775726, 0.36255223, 0.35726893, 0.35191009, 0.34647607, 0.3409673 , 0.33538426, 0.32972749, 0.32399761, 0.31819529, 0.31232133, 0.30637661, 0.30036211, 0.29427888, 0.2881265 , 0.28190832, 0.27562602, 0.26928147, 0.26287683, 0.25641457, 0.24989748, 0.24332878, 0.23671214, 0.23005179, 0.22335258, 0.21662012, 0.20986086, 0.20308229, 0.19629307, 0.18950326, 0.18272455, 0.17597055, 0.16925712, 0.16260273, 0.15602894, 0.14956101, 0.14322828, 0.13706449, 0.13110864, 0.12540538, 0.12000532, 0.11496505, 0.11034678, 0.10621724, 0.1026459 , 0.09970219, 0.09745186, 0.09595277, 0.09525046, 0.09537439, 0.09633538, 0.09812496, 0.1007168 , 0.10407067, 0.10813094, 0.11283773, 0.11812832, 0.12394051, 0.13021494, 0.13689671, 0.1439362 };

			float totalDist = m_maxScalarFieldVal - m_minScalarFieldVal;
			float wavelength = ((scalarValue - m_minScalarFieldVal) / totalDist) * 255;
			int pos = int(wavelength);
			if (totalDist < 0.0001) {
				pos = 0;
			}
			Vector3 color;
			color.x = rColorVec[pos];
			color.y = gColorVec[pos];
			color.z = bColorVec[pos];
			return color;
		}

		template<>
		Vector3 ScalarFieldRenderer<Vector2>::jetColorMap(Scalar scalarValue) {
			Vector3 color;
			return color;
		}

		template<>
		Vector3 ScalarFieldRenderer<Vector2>::grayScaleColorMap(Scalar scalarValue) {
			Vector3 color;
			return color;
		}
		template<>
		void ScalarFieldRenderer<Vector2>::updateMinMaxScalarField(const BaseWindow::scalarVisualization_t &visualizationType /* = BaseWindow::drawPressure */) const {
			Scalar maxValue, minValue;
			if (visualizationType == BaseWindow::scalarVisualization_t::drawFineGridScalars) {
				findMinMax(*m_pFineGridScalarField2D, minValue, maxValue);
				m_minScalarFieldVal = minValue; m_maxScalarFieldVal = maxValue;
			} else if(visualizationType != BaseWindow::scalarVisualization_t::drawNoScalarField) {
				findMinMax(RenderingUtils::getInstance()->switchScalarField2D(visualizationType, m_pGrid->getGridData2D()), minValue, maxValue);
				m_minScalarFieldVal = minValue; m_maxScalarFieldVal = maxValue;
			}
		}

		template<>
		void ScalarFieldRenderer<Vector3>::updateMinMaxScalarField(const BaseWindow::scalarVisualization_t &visualizationType /* = BaseWindow::drawPressure */) const {
			Scalar maxValue, minValue;
			if (visualizationType == BaseWindow::scalarVisualization_t::drawFineGridScalars) {
				findMinMax(*m_pFineGridStreamfunction3D, minValue, maxValue);
				m_minScalarFieldVal = minValue; m_maxScalarFieldVal = maxValue;
			}
			else if (visualizationType != BaseWindow::scalarVisualization_t::drawNoScalarField) {
				findMinMax(RenderingUtils::getInstance()->switchScalarField3D(visualizationType, m_pGrid->getGridData3D()), minValue, maxValue);
				m_minScalarFieldVal = minValue; m_maxScalarFieldVal = maxValue;
			}
		}

		/************************************************************************/
		/* Drawing                                                              */
		/************************************************************************/
		template <>
		void ScalarFieldRenderer<Vector2>::beginDrawScalarField(BaseWindow::scalarVisualization_t visualizationType, dimensions_t kthSlices /* = -1 */) {
			if (visualizationType == BaseWindow::scalarVisualization_t::drawFineGridScalars) {

			}
			if(m_updateScalarMinMax)
				updateMinMaxScalarField(visualizationType);
			
			updateValueColor(visualizationType);

			if (visualizationType == BaseWindow::scalarVisualization_t::drawFineGridScalars) {
				
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

				glEnableClientState(GL_VERTEX_ARRAY);
				glEnableClientState(GL_COLOR_ARRAY);
				glBindBuffer(GL_ARRAY_BUFFER, *m_pFineGridVerticesVBO);
				glVertexPointer(2, GL_FLOAT, 0, 0);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pFineGridIndexVBO);
				glBindBuffer(GL_ARRAY_BUFFER, *m_pFineGridColorsVBO);
				glColorPointer(3, GL_FLOAT, 0, 0);

				glDrawElements(GL_QUADS, 4 * (m_pFineGridScalarField2D->getDimensions().x - 1)*(m_pFineGridScalarField2D->getDimensions().y - 1), GL_UNSIGNED_INT, 0);

				glBindBuffer(GL_ARRAY_BUFFER, 0);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
				glDisableClientState(GL_VERTEX_ARRAY);
				glDisableClientState(GL_COLOR_ARRAY);

				if (m_drawFineGridCells) {
					glLineWidth(1.0f);
					glColor3f(0.0f, 0.0f, 0.0f);
					glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

					glEnableClientState(GL_VERTEX_ARRAY);
					glDisableClientState(GL_COLOR_ARRAY);
					glBindBuffer(GL_ARRAY_BUFFER, *m_pFineGridVerticesVBO);
					glVertexPointer(2, GL_FLOAT, 0, 0);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pFineGridIndexVBO);

					glDrawElements(GL_QUADS, 4 * (m_pFineGridScalarField2D->getDimensions().x - 1)*(m_pFineGridScalarField2D->getDimensions().y - 1), GL_UNSIGNED_INT, 0);

					glBindBuffer(GL_ARRAY_BUFFER, 0);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
					glDisableClientState(GL_VERTEX_ARRAY);
					glDisableClientState(GL_COLOR_ARRAY);
				}
			} else {
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

				glEnableClientState(GL_VERTEX_ARRAY);
				glEnableClientState(GL_COLOR_ARRAY);
				glBindBuffer(GL_ARRAY_BUFFER, *m_pGridCentroidsVBO);
				glVertexPointer(2, GL_FLOAT, 0, 0);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pScalarIndexXYVBO);
				glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarColorsVBO);
				glColorPointer(3, GL_FLOAT, 0, 0);

				if (m_pGrid->isPeriodic())
					glDrawElements(GL_QUADS, 4 * (m_gridDimensions.x)*(m_gridDimensions.y - 2), GL_UNSIGNED_INT, 0);
				else if (m_triangulateQuads)
					glDrawElements(GL_TRIANGLES, 3 * 4 * (m_gridDimensions.x - 1)*(m_gridDimensions.y - 1), GL_UNSIGNED_INT, 0);
				else
					glDrawElements(GL_QUADS, 4 * (m_gridDimensions.x - 1)*(m_gridDimensions.y - 1), GL_UNSIGNED_INT, 0);

				glBindBuffer(GL_ARRAY_BUFFER, 0);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
				glDisableClientState(GL_VERTEX_ARRAY);
				glDisableClientState(GL_COLOR_ARRAY);
			}
			if (m_pCutCells) {
				drawCutCellsScalarField(visualizationType);
			}
		}
		template <>
		void ScalarFieldRenderer<Vector2>::drawCutCellsScalarField(BaseWindow::scalarVisualization_t visualizationType) {
			glEnable(GL_STENCIL_TEST);
			glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
			glStencilFunc(GL_ALWAYS, 0, 1);
			glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT);
			glStencilMask(1);

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			for (int i = 0; i < m_pCutCells->getNumberCutCells(); i++) {
				auto cutcell = m_pCutCells->getCutCell(i);
				Vector2 cellCentroid = cutcell.getCentroid();
				
				Vector3 currColor;
				if (m_colorScheme == viridis) {
					currColor = viridisColorMap(m_pCutCellsScalarField->at(i));
				} else if (m_colorScheme == jet) {
					currColor = jetColorMap(m_pCutCellsScalarField->at(i));
				}
				if (m_colorScheme == grayscale) {
					currColor = grayScaleColorMap(m_pCutCellsScalarField->at(i));
				}
				glColor3f(currColor.x, currColor.y, currColor.z);
				glBegin(GL_TRIANGLE_FAN);
				glVertex2f(cellCentroid.x, cellCentroid.y);
				for (int j = 0; j < cutcell.getHalfEdges().size(); j++) {
					const Vector2 &vertexPos = cutcell.getHalfEdges()[j]->getVertices().first->getPosition();
					glVertex2f(vertexPos.x, vertexPos.y);
				}
				const Vector2 &iniPos = cutcell.getHalfEdges().front()->getVertices().first->getPosition();
				glVertex2f(iniPos.x, iniPos.y);
				glEnd();
				//glDisable(GL_STENCIL_TEST);
			}
			glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
			glStencilFunc(GL_EQUAL, 1, 1);
			glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

			for (int i = 0; i  < m_pCutCells->getNumberCutCells(); i++) {
				auto cutcell = m_pCutCells->getCutCell(i);
				Vector2 cellCentroid = cutcell.getCentroid();

				Vector3 currColor;
				if (m_colorScheme == viridis) {
					currColor = viridisColorMap(m_pCutCellsScalarField->at(i));
				}
				else if (m_colorScheme == jet) {
					currColor = jetColorMap(m_pCutCellsScalarField->at(i));
				}
				if (m_colorScheme == grayscale) {
					currColor = grayScaleColorMap(m_pCutCellsScalarField->at(i));
				}
				glColor3f(currColor.x, currColor.y, currColor.z);
				glBegin(GL_TRIANGLE_FAN);
				
				glVertex2f(cellCentroid.x, cellCentroid.y);
				for (int j = 0; j < cutcell.getHalfEdges().size(); j++) {
					const Vector2 &vertexPos = cutcell.getHalfEdges()[j]->getVertices().first->getPosition();
					glVertex2f(vertexPos.x, vertexPos.y);
				}
				const Vector2 &iniPos = cutcell.getHalfEdges().front()->getVertices().first->getPosition();
				glVertex2f(iniPos.x, iniPos.y);
				glEnd();
				//glDisable(GL_STENCIL_TEST);
			}
			glDisable(GL_STENCIL_TEST);
		}
		

		template <>
		void ScalarFieldRenderer<Vector3>::beginDrawScalarField(BaseWindow::scalarVisualization_t visualizationType, dimensions_t kthSlices) {
			Scalar dx = m_pGrid->getGridData3D()->getScaleFactor(0, 0, 0).x;
			if(m_updateScalarMinMax)
				updateMinMaxScalarField(visualizationType);

			updateValueColor(visualizationType);

			
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

			glEnableClientState(GL_VERTEX_ARRAY);                 
			glEnableClientState(GL_COLOR_ARRAY);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pGridCentroidsVBO);
			glVertexPointer(3, GL_FLOAT, 0, 0);	
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pScalarIndexXYVBO);
			glBindBuffer(GL_ARRAY_BUFFER, *m_pScalarColorsVBO);
			glColorPointer(3, GL_FLOAT, 0, 0);

			int kthSlice = clamp(kthSlices.z, 0, m_gridDimensions.z);
			//XY
			size_t initialIndex = (m_gridDimensions.x - 1)*(m_gridDimensions.y - 1)*kthSlice * 4 * sizeof(int);
			glPushMatrix();
			glTranslatef(0, 0, -(0.5*dx));
			glDrawElements(GL_QUADS, 4*(m_gridDimensions.x - 1)*(m_gridDimensions.y - 1), GL_UNSIGNED_INT, (void *) initialIndex);
			glPopMatrix();

			//YZ
			kthSlice = clamp(kthSlices.x, 0, m_gridDimensions.x);
			initialIndex = (m_gridDimensions.y - 1)*(m_gridDimensions.z - 1)*kthSlice * 4 * sizeof(int);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pScalarIndexYZVBO);
			glPushMatrix();
			glTranslatef(-(0.5*dx), 0, 0);
			glDrawElements(GL_QUADS, 4 * (m_gridDimensions.y - 1)*(m_gridDimensions.z - 1), GL_UNSIGNED_INT, (void *)initialIndex);
			glPopMatrix();
			
			//XZ
			kthSlice = clamp(kthSlices.y, 0, m_gridDimensions.y);
			initialIndex = (m_gridDimensions.x - 1)*(m_gridDimensions.z - 1)*kthSlice * 4 * sizeof(int);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *m_pScalarIndexXZVBO);
			glPushMatrix();
			glTranslatef(0, -(0.5*dx), 0);
			glDrawElements(GL_QUADS, 4 * (m_gridDimensions.x - 1)*(m_gridDimensions.z - 1), GL_UNSIGNED_INT, (void *)initialIndex);
			glPopMatrix();

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_COLOR_ARRAY);
		}

		template <class VectorT>
		void ScalarFieldRenderer<VectorT>::endDrawScalarField() const {
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glDisableClientState(GL_COLOR_ARRAY);
		}
		
		/************************************************************************/
		/* FVRenderer declarations - Linking time                               */
		/************************************************************************/
		template ScalarFieldRenderer<Vector2>;
		template ScalarFieldRenderer<Vector3>;
	}
}