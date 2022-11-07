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

#ifndef __CAMERA_H_
#define __CAMERA_H_

#include "ChimeraCore.h"
#include "ChimeraGrids.h"


namespace Chimera {

	using namespace Core;
	using namespace Grids;

	namespace Rendering {

		typedef enum cameraTypes_t {
			orthogonal2D,
			perspective3D
		} cameraTypes_t;

		typedef enum camActions {
			moving,
			zooming,
			following,
			rotatingAroundGrid,
			none
		} camAction;

		class Camera {

		public:

			/************************************************************************/
			/* ctors and dtors					                                    */
			/************************************************************************/
			//Default constructor
			Camera(cameraTypes_t camType);

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			Scalar* getModelview() const {
				return (Scalar *) m_modelView;
			}

			Scalar getRotAxisY() const {
				return m_rotY;
			}

			Scalar getRotAxisX() const {
				return m_rotX;
			}

			const Vector3 & getUpAxis() const {
				return m_up;
			}

			const Vector3 & getPosition() const {
				return m_position;
			}

			void setPosition(const Vector3 &camPosition) {
				m_position = camPosition;
			}

			const Vector3 & getDirection() const {
				return m_direction;
			}

			void setDirection(const Vector3 &camDirection) {
				m_direction = camDirection;
			}

			cameraTypes_t getType() const {
				return m_camType;
			}

			void setType(cameraTypes_t camType) {
				m_camType = camType;
			}

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/

			/**A keyboard callback, that is util when assigning keyboard entries for
			rendering parameters modifications.
			@param key: the pushed key*/
			void keyboardCallback(unsigned char key, int x, int y);

			/**A mouse callback, that is util when assigning mouse callbacks for
			rendering parameters modifications.
			@param button: mouse button ID
			@param state: mouse button state
			@param x: mouse poisition according to the x axis.
			@param y: mouse poisition according to the x axis. */
			void mouseCallback(int button, int state, int x, int y);

			/**A mouse motion callback, that is util when assigning mouse callbacks for
			rendering parameters modifications.
			@param x: mouse poisition according to the x axis.
			@param y: mouse poisition according to the x axis. */
			void motionCallback(int x, int y);

			/** Reshape callback*/
			void reshapeCallback(int width, int height);

			/** Update the camera within time*/
			void update(Scalar dt);

			/** Updates the GL context with the camera info*/
			void updateGL();

			inline void followGrid(QuadGrid *pQuadGrid, const Vector3 &followOffset) {
				m_pFollowGrid = pQuadGrid;
				m_camAction = following;
				m_followOffset = followOffset;
			}

			inline void setFixedDirection(bool fixed) {
				m_fixedDirection = fixed;
			}

			inline void setRotationAroundGridMode(const Vector3 &lookAtPoint) {
				m_camAction = rotatingAroundGrid;
				m_lookAtPoint = lookAtPoint;
			}


			Vector3 getOGLPos(int x, int y);

			Vector2 getWorldMousePosition();
		private:

			/************************************************************************/
			/* Private members                                                      */
			/************************************************************************/
			cameraTypes_t m_camType;
			/** The camera position. */
			Vector3	m_position;
			/** The camera direction. */
			Vector3 m_direction;
			/** The camera up vector. */
			Vector3	m_up;

			Vector3 m_velocity;

			/** The openGL window width */
			Scalar	m_windowWidth;
			/** The openGL window height */
			Scalar	m_windowHeight;

			Scalar m_modelView[16];

			Scalar nearClippingPane;
			Scalar farClippingPane;

			Scalar m_mouseMinSensivity;
			Scalar m_mouseMaxSensivity;
			Scalar m_mouseSensivity;
			camAction m_camAction;
			//For additional functionalities
			camAction m_lastCamAction;

			/** Mouse X coordinates */
			int		m_mouseX;
			/** Mouse Y coordinates */
			int		m_mouseY;

			Scalar m_rotX;
			Scalar m_rotY;




			Vector3 m_initialPosition;

			//Following parameters
			bool m_fixedDirection;
			QuadGrid *m_pFollowGrid;
			Vector3 m_followOffset;

			/** Rotating around point parameters */
			Vector3 m_lookAtPoint;

		};
	}
}

#endif
