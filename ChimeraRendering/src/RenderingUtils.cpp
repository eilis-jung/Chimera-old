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

#include "RenderingUtils.h"
#include "Windows/BaseWindow.h"

namespace Chimera {
	
	using namespace Windows;

	namespace Rendering {
		

		template<class VectorType>
		void RenderingUtils::drawVectorT(const VectorType &startingPoint, const VectorType &endingPoint, isVector2True, Scalar arrowLenght) {
			glDisable(GL_LIGHTING);

			//glLineWidth(10.0f);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glBegin(GL_LINES);
				glVertex2f(startingPoint.x, startingPoint.y);
				glVertex2f(endingPoint.x, endingPoint.y);
			glEnd();

			VectorType normalizedVel = (endingPoint - startingPoint);
			Scalar lenght = normalizedVel.length();
			normalizedVel.normalize();

			Scalar finalLenght = clamp(arrowLenght*log(1 + lenght*10), 0.001f, 0.008f);
			
			VectorType a0 = normalizedVel*finalLenght*0.866 + endingPoint;
			VectorType a1 = normalizedVel.perpendicular()*finalLenght*0.5 + endingPoint;
			VectorType a2 = -normalizedVel.perpendicular()*finalLenght*0.5 + endingPoint;
			
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glBegin(GL_TRIANGLES);
				glVertex2f(a0.x, a0.y);
				glVertex2f(a1.x, a1.y);
				glVertex2f(a2.x, a2.y);
			glEnd();
		}

		template<class VectorType>
		void RenderingUtils::drawVectorT(const VectorType &startingPoint, const VectorType &endingPoint, isVector2False, Scalar arrowLenght) {
			glDisable(GL_LIGHTING);

			glLineWidth(0.5f);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glColor3f(0.0f, 0.0f, 0.0f);
			glBegin(GL_LINES);
			glVertex3f(startingPoint.x, startingPoint.y, startingPoint.z);
			glVertex3f(endingPoint.x, endingPoint.y, endingPoint.z);
			glEnd();
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


			VectorType normalizedVel = (endingPoint - startingPoint);
			Scalar lenght = normalizedVel.length();
			normalizedVel.normalize();

			Scalar finalLenght = clamp(arrowLenght*log(1 + lenght * 10), 0.001f, 0.008f);

			VectorType a0 = normalizedVel*finalLenght*0.866 + endingPoint;
			VectorType a1 = normalizedVel.cross(VectorType(0, 1, 0))*finalLenght*0.5 + endingPoint;
			VectorType a2 = -normalizedVel.cross(VectorType(0, 1, 0))*finalLenght*0.5 + endingPoint;

			if (a1.length() > 0) {
				glBegin(GL_TRIANGLES);
				glVertex3f(a0.x, a0.y, a0.z);
				glVertex3f(a1.x, a1.y, a1.z);
				glVertex3f(a2.x, a2.y, a2.z);
				glEnd();
			}
			a1 = normalizedVel.cross(VectorType(1, 0, 0))*finalLenght*0.5 + endingPoint;
			a2 = -normalizedVel.cross(VectorType(1, 0, 0))*finalLenght*0.5 + endingPoint;

			if (a1.length() > 0) {
				glBegin(GL_TRIANGLES);
				glVertex3f(a0.x, a0.y, a0.z);
				glVertex3f(a1.x, a1.y, a1.z);
				glVertex3f(a2.x, a2.y, a2.z);
				glEnd();
			}

			a1 = normalizedVel.cross(VectorType(0, 0, 1))*finalLenght*0.5 + endingPoint;
			a2 = -normalizedVel.cross(VectorType(0, 0, 1))*finalLenght*0.5 + endingPoint;

			if (a1.length() > 0) {
				glBegin(GL_TRIANGLES);
				glVertex3f(a0.x, a0.y, a0.z);
				glVertex3f(a1.x, a1.y, a1.z);
				glVertex3f(a2.x, a2.y, a2.z);
				glEnd();
			}
		}


		void RenderingUtils::drawVectorArrow(const Vector2 &vectorArrow, Scalar arrowSize) {
			Vector2 normalizedVec = (vectorArrow).normalized();
			Scalar vecSize = vectorArrow.length();
			Scalar finalTriangleLength = 0;
			finalTriangleLength = clamp(arrowSize*log(1 + vecSize*10), 0.001f, 0.008f);


			//Calculate v1
			Vector2 arrowVertices[3];
			arrowVertices[0] = normalizedVec.perpendicular();
			arrowVertices[0] *= finalTriangleLength/2;

			//Calculate v2
			arrowVertices[1] = -normalizedVec.perpendicular();
			arrowVertices[1] *= finalTriangleLength/2;

			//Calculate v3
			arrowVertices[2] *= normalizedVec*0.866*finalTriangleLength;

			glBegin(GL_TRIANGLES);
				glVertex2f(arrowVertices[0].x, arrowVertices[0].y);
				glVertex2f(arrowVertices[1].x, arrowVertices[1].y);
				glVertex2f(arrowVertices[2].x, arrowVertices[2].y);
			glEnd();
		}

		void RenderingUtils::drawVectorArrow(const Vector3 &vectorArrow, Scalar arrowSize) {
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			Vector3 tempVec = vectorArrow;
			Vector3 normalizedVec = tempVec.normalized();
			Scalar vecSize = vectorArrow.length();
			Scalar finalTriangleLength = 0;
			finalTriangleLength = clamp(arrowSize*log(1 + vecSize*10), 0.001f, 0.008f);


			//Calculate v1
			Vector3 arrowVertices[3];
			arrowVertices[0] = normalizedVec.cross(Vector3(1, 0, 0));
			arrowVertices[0] *= finalTriangleLength/2;

			//Calculate v2
			arrowVertices[1] = -normalizedVec.cross(Vector3(-1, 0, 0));
			arrowVertices[1] *= finalTriangleLength/2;

			//Calculate v3
			arrowVertices[2] *= normalizedVec*0.866*finalTriangleLength;

			glBegin(GL_TRIANGLES);
			glVertex2f(arrowVertices[0].x, arrowVertices[0].y);
			glVertex2f(arrowVertices[1].x, arrowVertices[1].y);
			glVertex2f(arrowVertices[2].x, arrowVertices[2].y);
			glEnd();
		}
		

		const Array2D<Scalar> &  RenderingUtils::switchScalarField2D(const BaseWindow::scalarVisualization_t &visualizationType, Grids::GridData2D * pGridData2D) {
			switch(visualizationType) {
				case BaseWindow::drawPressure:
					return pGridData2D->getPressureArray();
					break;

				case BaseWindow::drawDensityField:
					return *pGridData2D->getDensityBuffer().getBufferArray1();
					break;

				case BaseWindow::drawTemperature:
					return *pGridData2D->getTemperatureBuffer().getBufferArray1();
					break;

				case BaseWindow::drawVorticity:
					return pGridData2D->getVorticityArray();
					break;

				case BaseWindow::drawLevelSet:
					return pGridData2D->getLevelSetArray();
					break;

				case BaseWindow::drawDivergent:
					return pGridData2D->getDivergentArray();
					break;

				case BaseWindow::drawStreamfunction:
					return pGridData2D->getStreamfunctionArray();
				break;

				case BaseWindow::drawKineticEnergy: 
					return pGridData2D->getKineticEnergyArray();
					break;

				case BaseWindow::drawKineticEnergyChange:
					return pGridData2D->getKineticEnergyChangeArray();
					break;
			}
			
		}

		const Array3D<Scalar> &  RenderingUtils::switchScalarField3D(const BaseWindow::scalarVisualization_t &visualizationType, Grids::GridData3D * pGridData3D) {
			switch(visualizationType) {
			case BaseWindow::drawPressure:
				return pGridData3D->getPressureArray();
				break;

			case BaseWindow::drawDensityField:
				return *pGridData3D->getDensityBuffer().getBufferArray1();
				break;

			case BaseWindow::drawTemperature:
				return *pGridData3D->getTemperatureBuffer().getBufferArray1();
				break;

			case BaseWindow::drawVorticity:
				return pGridData3D->getVorticityArray();
				break;

			case BaseWindow::drawLevelSet:
				return pGridData3D->getLevelSetArray();
				break;

			case BaseWindow::drawDivergent:
				return pGridData3D->getDivergentArray();
				break;

			case BaseWindow::drawKineticEnergy:
				return pGridData3D->getKineticEnergyArray();
				break;

			case BaseWindow::drawKineticEnergyChange:
				return pGridData3D->getKineticEnergyChangeArray();
				break;
			}
		}


		template void RenderingUtils::drawVectorT<Vector2>(const Vector2 &startingPoint, const Vector2 &endingPoint, isVector2True, Scalar arrowLenght);
		template void RenderingUtils::drawVectorT<Vector2D>(const Vector2D &startingPoint, const Vector2D &endingPoint, isVector2True, Scalar arrowLenght);

		template void RenderingUtils::drawVectorT<Vector3>(const Vector3 &startingPoint, const Vector3 &endingPoint, isVector2False, Scalar arrowLenght);
		template void RenderingUtils::drawVectorT<Vector3D>(const Vector3D &startingPoint, const Vector3D &endingPoint, isVector2False, Scalar arrowLenght);
	}

}