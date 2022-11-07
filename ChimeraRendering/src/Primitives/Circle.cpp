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


#include "Primitives/Circle.h"


namespace Chimera {

	namespace Rendering {

		#pragma region Constructors
		template <>
		Circle<Vector2>::Circle(const Vector2 &position, const Vector2 &normal, Scalar size, int numSubdivisions /* = 32 */) :
			PhysicalObject<Vector2>(position, normal, Vector2(1, 1)) {
			m_circlePoints.push_back(m_position);

			for(int i = 0; i < numSubdivisions + 2; i++) {
				Scalar angle = (i + 1)*(360/numSubdivisions);
				Vector2 vertexPosition;
				vertexPosition.x = m_position.x + size*cos(DegreeToRad(angle));
				vertexPosition.y = m_position.y + size*sin(DegreeToRad(angle));
				m_circlePoints.push_back(vertexPosition);
			}
		}

		template <>
		Circle<Vector3>::Circle(const Vector3 &position, const Vector3 &normal, Scalar size, int numSubdivisions /* = 32 */) :
			PhysicalObject<Vector3>(position, normal, Vector3(1, 1, 1)) {
			/*m_circlePoints.push_back(m_position);

			for (int i = 0; i < numSubdivisions + 2; i++) {
				Scalar angle = (i + 1)*(360 / numSubdivisions);
				Vector3 vertexPosition;
				vertexPosition.x = m_position.x + size*cos(DegreeToRad(angle));
				vertexPosition.y = m_position.y + size*sin(DegreeToRad(angle));
				m_circlePoints.push_back(vertexPosition);
			}*/
		}
		#pragma endregion

		#pragma region Functionalities
		template<>
		void Circle<Vector2>::draw() {
			glPushMatrix();
			glColor3f(0.8f, 0.1f, 0.05f);
			glBegin(GL_TRIANGLE_FAN);
				for(int i = 0; i < m_circlePoints.size(); i++) {
					glVertex2f(m_circlePoints[i].x, m_circlePoints[i].y);
				}
			glEnd();
			glPopMatrix();
		}

		template<>
		void Circle<Vector3>::draw() {
			glPushMatrix();
			glColor3f(0.8f, 0.1f, 0.05f);
			glBegin(GL_TRIANGLE_FAN);
			for (int i = 0; i < m_circlePoints.size(); i++) {
				glVertex3f(m_circlePoints[i].x, m_circlePoints[i].y, m_circlePoints[i].z);
			}
			glEnd();
			glPopMatrix();
		}
		#pragma endregion

		template class Circle<Vector2>;
		template class Circle<Vector3>;
	}
}