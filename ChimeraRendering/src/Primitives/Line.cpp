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

#include "Primitives/Line.h"


namespace Chimera {
	namespace Rendering {

		#pragma region Constructors
		template <class VectorT>
		Line<VectorT>::Line(const VectorT &position, const vector<VectorT> &points) : PhysicalObject(position, VectorT(), VectorT()) {
			m_linePoints = points;
			m_lineWidth = 3.0f;
		}

		template <class VectorT>
		Line<VectorT>::Line(const VectorT &position, string segmentFile ) : PhysicalObject(position, VectorT(), VectorT()) {
			m_lineWidth = 3.0f;
			shared_ptr<ifstream> fileStream(new ifstream(segmentFile.c_str()));
			if(fileStream->fail())
				throw("File not found: " + segmentFile);

			Scalar temp;
			Vector2 currPoint;
			Vector2 position2D = Vector2(m_position.x, m_position.y);
			int numPoints;
			(*fileStream) >> numPoints;
			for(int i = 0; i < numPoints; i++) {
				(*fileStream) >> currPoint.x;
				(*fileStream) >> currPoint.y;
				(*fileStream) >> temp;
				m_linePoints.push_back(currPoint + position2D);
			}
		}
		#pragma endregion 
		
		#pragma region Functionalities
		template <>
		void Line<Vector2>::draw() {
			glPushMatrix();
			glColor3f(0.0f, 0.0f, 0.0f);
			glLineWidth(m_lineWidth);
			glBegin(GL_LINES);
			for(int i = 0; i < m_linePoints.size() - 1; i++) {
				glVertex2f(m_linePoints[i].x, m_linePoints[i].y);
				glVertex2f(m_linePoints[i + 1].x, m_linePoints[i + 1].y);
			}
			glEnd();

			glPointSize(4.0f);
			glColor3f(0.0f, 1.0f, 0.0f);
			glBegin(GL_POINTS);
			for(int i = 0; i < m_linePoints.size(); i++) {
				glVertex2f(m_linePoints[i].x, m_linePoints[i].y);
			}
			glEnd();
			glLineWidth(1.0f);
			glPopMatrix();
		}

		template <>
		void Line<Vector3>::draw() {
			glPushMatrix();
			glColor3f(0.0f, 0.0f, 0.0f);
			glLineWidth(m_lineWidth);
			glBegin(GL_LINES);
			for (int i = 0; i < m_linePoints.size() - 1; i++) {
				glVertex3f(m_linePoints[i].x, m_linePoints[i].y, m_linePoints[i].z);
				glVertex3f(m_linePoints[i + 1].x, m_linePoints[i + 1].y, m_linePoints[i + 1].z);
			}
			glEnd();

			glPointSize(4.0f);
			glColor3f(0.0f, 1.0f, 0.0f);
			glBegin(GL_POINTS);
			for (int i = 0; i < m_linePoints.size(); i++) {
				glVertex3f(m_linePoints[i].x, m_linePoints[i].y, m_linePoints[i].z);
			}
			glEnd();
			glLineWidth(1.0f);
			glPopMatrix();
		}
		#pragma endregion 

		template class Line<Vector2>;
		template class Line<Vector3>;
	}
}