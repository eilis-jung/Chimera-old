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

#include "Primitives/Camera.h"

namespace Chimera {
	/************************************************************************/
	/* ctors and dtors                                                      */
	/************************************************************************/
	namespace Rendering {

		Camera::Camera(cameraTypes_t camType) : m_camType(camType) {

			m_position = m_initialPosition = Vector3(0, 1.5, 4);
			m_up = Vector3(0, 1, 0); 
			m_direction = Vector3(0, 0, -1);

			m_fixedDirection = false;

			m_mouseX = 0;
			m_mouseY = 0;
			m_mouseSensivity = 0.007;
			m_mouseMaxSensivity = 0.01;
			m_mouseMinSensivity = 0.00001;

			m_windowWidth = 800;
			m_windowHeight = 600;

			nearClippingPane = 0.001f;
			farClippingPane = 45.0f;

			for(int i = 0; i < 16; i++)
				m_modelView[i] = 0;

			m_modelView[0] = 1.0f;
			m_modelView[5] = 1.0f;
			m_modelView[10] = 1.0f;
			m_modelView[15] = 1.0f;

			m_rotX = m_rotY = 0;

			m_pFollowGrid = NULL;

			m_camAction = m_lastCamAction = none;
		}

		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		Vector3 Camera::getOGLPos(int x, int y) {
			GLint viewport[4];
			GLdouble modelview[16];
			GLdouble projection[16];
			GLfloat winX, winY, winZ;
			GLdouble posX, posY, posZ;

			glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
			glGetDoublev( GL_PROJECTION_MATRIX, projection );
			glGetIntegerv( GL_VIEWPORT, viewport );

			winX = (float)x;
			winY = (float)viewport[3] - (float)y;
			glReadPixels( x, int(winY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ );

			gluUnProject( winX, winY, winZ, modelview, projection, viewport, &posX, &posY, &posZ);

			return Vector3(posX, posY, posZ);
		}

		Vector2 Camera::getWorldMousePosition() {
			Vector3 worldPos3 = getOGLPos(m_mouseX, m_mouseY);
			return Vector2(worldPos3.x, worldPos3.y);
		}

		void Camera::keyboardCallback(unsigned char key, int x, int y) {
			Vector3 centeredPosition;
			switch(m_camType) {
				case orthogonal2D:
					switch(key) {
						//Camera movement:
						case 'a': case 'A': m_position.x += 0.05; break;
						case 'd': case 'D': m_position.x -= 0.05; break;
						case 'w': case 'W':	m_position.y += 0.05; break;
						case 's': case 'S':	m_position.y -= 0.05; break;
						case 't': case 'T': m_position += m_direction*0.05; break;
						case 'y': case 'Y': m_position -= m_direction*0.05; break;
						case 'c': case 'C': Logger::get() << "Camera position: (" << m_position.x << ", " 
																					<< m_position.y << ", " 
																					<< m_position.z << ")" << endl;
					}
				break;

				case perspective3D:
					Vector3 right = m_direction.cross(Vector3(0, 1, 0));
					right.normalize();
					switch(key) {
						//Camera movement:
						case 'w': case 'W':	m_position += m_direction*0.001; break;
						case 's': case 'S':	m_position -= m_direction*0.001; break;
						case 'a': case 'A':	
							//m_position -= right*0.001; 
							centeredPosition = m_position - m_lookAtPoint;
							Quaternion(Vector3(0, 1, 0), DegreeToRad(-15.0) * 3).rotate(&centeredPosition);
							m_position = centeredPosition + m_lookAtPoint;
							m_direction = (m_lookAtPoint - m_position).normalized();
						break;
						case 'd': case 'D':	//m_position += right*0.001; 
							centeredPosition = m_position - m_lookAtPoint;
							Quaternion(Vector3(0, 1, 0), DegreeToRad(15.0) * 3).rotate(&centeredPosition);
							m_position = centeredPosition + m_lookAtPoint;
							m_direction = (m_lookAtPoint - m_position).normalized();
						break;
						case 'f': case 'F': case 'm': case 'M': m_lastCamAction = m_camAction = moving; break;
						case 'r': case 'R': m_lastCamAction = m_camAction = rotatingAroundGrid; break;
					}
				break;
			}

			
		}


		void Camera::mouseCallback(int button, int state, int x, int y) {
			if(m_camType == orthogonal2D && state == GLUT_DOWN) {
				if(button == GLUT_MIDDLE_BUTTON) {
					m_lastCamAction = m_camAction;
					m_camAction = moving; 
				} else if(button == GLUT_RIGHT_BUTTON) {
					m_lastCamAction = m_camAction;
					m_camAction = zooming;
				} else if(button == GLUT_LEFT) {
					m_lastCamAction = m_camAction;
					m_camAction = none;
				}
			} else if(m_camType == orthogonal2D && state == GLUT_UP) {
				if(m_lastCamAction == following)
					m_camAction = following;
			}
			if (m_camType == perspective3D && state == GLUT_DOWN) {
				if (button == GLUT_RIGHT_BUTTON) {
					m_lastCamAction = m_camAction;
					m_camAction = zooming;
				}
			}
			else if (m_camType == perspective3D && state == GLUT_UP) {
				if (button == GLUT_RIGHT_BUTTON) {
					if (m_lastCamAction == rotatingAroundGrid)
						m_camAction = rotatingAroundGrid;
				}
			}
			m_mouseX = x;
			m_mouseY = y;
		}


		void Camera::motionCallback(int x, int y) {
			m_rotY = (m_mouseX - x);
			m_rotX = (m_mouseY - y);
			Scalar mouseSensitivity = clamp(m_mouseSensivity*m_position.z*m_position.z, m_mouseMinSensivity, m_mouseMaxSensivity);
			m_direction.normalize();
			m_up = m_direction.cross(Vector3(0,1,0));
			Vector3 crossLeft = m_direction.cross(m_up);
			crossLeft.normalize();
			Vector3 centeredPosition;
			switch(m_camType) {
				case orthogonal2D:
					switch(m_camAction) {
						case moving:
							/*m_position -= crossLeft*m_rotX*mouseSensitivity;
							m_position += m_up*m_rotY*mouseSensitivity;*/
							m_position.x += m_rotY*mouseSensitivity;
							m_position.y += m_rotX*mouseSensitivity;
							break;

						case zooming:
							m_position -= m_direction*m_rotX*mouseSensitivity;
							break;

						case none:
							break;
					}
				break;

				case perspective3D:
					switch (m_camAction) {
						case rotatingAroundGrid:
							centeredPosition = m_position - m_lookAtPoint;
							Quaternion(Vector3(0, 1, 0), DegreeToRad(m_rotY)*3).rotate(&centeredPosition);
							Quaternion(m_up,		 DegreeToRad(m_rotX)*4).rotate(&centeredPosition);
							m_position = centeredPosition + m_lookAtPoint;
							m_direction = (m_lookAtPoint - m_position).normalized();
						break;
						case zooming:
							m_position -= m_direction*m_rotX*0.01;
						break;
						default:
							Quaternion(Vector3(0, 1, 0), DegreeToRad(m_rotY)).rotate(&m_direction);
							Quaternion(m_up,			 DegreeToRad(m_rotX)).rotate(&m_direction);

							m_direction.normalize();
						break;
					}
				break;
			}
			m_mouseX = x;
			m_mouseY = y;
		}


		void Camera::updateGL() {
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			Scalar zx, zy;
			switch(m_camType) {
				case orthogonal2D:
					if(m_camAction == following) {
						m_position.x = m_pFollowGrid->getPosition().x + m_followOffset.x;
						m_position.y = m_pFollowGrid->getPosition().y + m_followOffset.y;
						m_position.x += (m_pFollowGrid->getGridCentroid().x - m_pFollowGrid->getGridOrigin().x);
						m_position.y += (m_pFollowGrid->getGridCentroid().y - m_pFollowGrid->getGridOrigin().y);
					}
					zx = m_windowWidth/(2*m_windowHeight);
					zy = 0.5;
					zx *= m_position.z;
					zy *= m_position.z;
					glOrtho(-zx, zx, -zy, zy, nearClippingPane, farClippingPane);
					gluLookAt(m_position.x, m_position.y, m_position.z, m_position.x + m_direction.x, m_position.y + m_direction.y, m_position.z + m_direction.z, 0.0f, 1.0f, 0.0f);
				break;

				case perspective3D:
					gluPerspective(60.0f, m_windowWidth/m_windowHeight, nearClippingPane, farClippingPane);
					/*if(m_camAction == following) {
						m_position = m_pFollowObject->getPosition() + m_followOffset;
						if(m_fixedDirection)
							gluLookAt(m_position.x, m_position.y, m_position.z, m_position.x + m_direction.x, m_position.y + m_direction.y, m_position.z + m_direction.z, 0.0f, 1.0f, 0.0f);
						else {
							m_direction = (m_pFollowObject->getPosition() - m_position).normalized();
							gluLookAt(m_position.x, m_position.y, m_position.z, m_pFollowObject->getPosition().x, m_pFollowObject->getPosition().y, m_pFollowObject->getPosition().z, 0.0f, 1.0f, 0.0f);
						}
					} else {*/
						gluLookAt(m_position.x, m_position.y, m_position.z, m_position.x + m_direction.x, m_position.y + m_direction.y, m_position.z + m_direction.z, 0.0f, 1.0f, 0.0f);
					//}

					
				break;
			}	
		}

		void Camera::reshapeCallback(int width, int height) {
			m_windowWidth = width;
			m_windowHeight = height;
		}

	}
	
}