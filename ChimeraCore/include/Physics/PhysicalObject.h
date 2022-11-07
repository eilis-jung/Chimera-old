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

#ifndef __RENDERING_OBJECT_3D_
#define __RENDERING_OBJECT_3D_

#include "Config/ChimeraConfig.h"

#include "Math/DoubleScalar.h"
#include "Math/Vector2.h"
#include "Math/Vector2d.h"
#include "Math/Vector3.h"
#include "Math/Vector3D.h"

using namespace std;
namespace Chimera {

	namespace Core {

		typedef enum positionUpdateType_t {
			sinFunction,
			cosineFunction,
			uniformFunction,
			pathAnimation,
			noPositionUpdate
		} positionUpdateType_t;

		typedef enum rotationType_t {
			constantRotation,
			alternatingRotation,
			noRotation
		} rotationFunctionType_t;

		typedef enum couplingType_t {
			oneWayCouplingSolidToFluid, //Solid dictates the velocity update of the fluid
			oneWayCouplingFluidToSolid, //Fluid dictates the velocity update of the solid
			twoWayCoupling
		} couplingType_t;

		template <class VectorT>
		class PhysicalObject {

		public:
			#pragma region ExternalStructures
			typedef struct positionUpdate_t {
				positionUpdateType_t positionUpdateType;
				Scalar amplitude;
				Scalar frequency;
				VectorT direction;

				Scalar startingTime;
				Scalar endingTime;

				bool absoluteValuesOnly;

				vector<VectorT> pathMesh;

				positionUpdate_t() {
					positionUpdateType = noPositionUpdate;
					amplitude = frequency = 0;
					startingTime = 0;
					endingTime = FLT_MAX;
					absoluteValuesOnly = true;
				}

				VectorT update(Scalar elapsedTime) {
					VectorT zeroVector;
					elapsedTime -= startingTime;
					if (elapsedTime < 0)
						return zeroVector;

					if (positionUpdateType == positionUpdateType_t::sinFunction) {
						DoubleScalar functionValue = sin(frequency*elapsedTime*PI)*amplitude / PI;
						return direction*functionValue;
					}
					else if (positionUpdateType == positionUpdateType_t::cosineFunction) {
						DoubleScalar functionValue = cos(frequency*elapsedTime*PI)*amplitude / PI;
						return direction*functionValue;
					}
					else if (positionUpdateType == positionUpdateType_t::uniformFunction) {
						return direction*amplitude*elapsedTime;
					}
					return zeroVector;
				}
			} positionUpdate_t;

			typedef struct rotationUpdate_t {
				rotationType_t rotationType;
				Scalar initialRotation;
				Scalar speed;
				Scalar acceleration;
				Scalar minAngle;
				Scalar maxAngle;

				VectorT axis;

				Scalar startingTime;
				Scalar endingTime;

				rotationUpdate_t() {
					initialRotation = 0;
					speed = acceleration = 0;
					startingTime = 0;
					minAngle = 0;
					maxAngle = 0;
					endingTime = FLT_MAX;
					rotationType = noRotation;
				}

				Scalar update2D(Scalar elapsedTime) {
					elapsedTime -= startingTime;
					if (elapsedTime < 0)
						return 0;

					if (rotationType == alternatingRotation) {
						Scalar rotationAngle = initialRotation + elapsedTime*speed + elapsedTime*elapsedTime*(acceleration) / 2;
						if (rotationType == alternatingRotation && (rotationAngle - initialRotation > maxAngle || rotationAngle - initialRotation < minAngle)) {
							speed = -speed;
							initialRotation = rotationAngle;
							startingTime += elapsedTime;
						}
						if (elapsedTime < endingTime - startingTime)
							return rotationAngle;
					}
					else if (rotationType == constantRotation) {
						if (elapsedTime < endingTime - startingTime)
							return initialRotation + elapsedTime*speed + elapsedTime*elapsedTime*(acceleration) / 2;
					} 
					return 0;
				}
			} rotationUpdate_t;
			#pragma endregion

			#pragma region Constructors
			FORCE_INLINE PhysicalObject(const VectorT & position, const VectorT &eulerAngles, const VectorT &scale) :
				m_position(position), m_eulerAngles(eulerAngles), m_scale(scale) {
				
			}
			#pragma endregion

			#pragma region Functionalities
			virtual void update(Scalar dt) = 0;
			virtual void draw() { }
			#pragma endregion

			#pragma region AccessFunctions
			const string & getName() const {
				return m_objectName;
			}
			void setName(const string &objectName) {
				m_objectName = objectName;
			}

			const VectorT & getPosition() const {
				return m_position;
			}

			void setPosition(const VectorT & position) {
				m_position = position;
			}

			const VectorT & getVelocity() const {
				return m_velocity;
			}

			void setVelocity(const VectorT & velocity) {
				m_velocity = velocity;
			}

			//Returns orientation in Euler Angles 
			const VectorT & getOrientation() const {
				return m_eulerAngles;
			}

			void setOrientation(const VectorT & orientation) {
				m_eulerAngles = orientation;
			}


			void setVelocityFunction(const positionUpdate_t &positionFunction) {
				m_positionUpdate = positionFunction;
			}

			void setRotationFunction(const rotationUpdate_t &rotationFunction) {
				m_rotationUpdate = rotationFunction;
			}

			positionUpdate_t & getVelocityFunction() {
				return m_positionUpdate;
			}

			const positionUpdate_t & getVelocityFunction() const {
				return m_positionUpdate;
			}

			rotationUpdate_t & getRotationFunction() {
				return m_rotationUpdate;
			}

			const rotationUpdate_t & getRotationFunction() const {
				return m_rotationUpdate;
			}

			VectorT getEffectiveVelocity(int timeOffset);
			#pragma endregion

		protected:

			#pragma region PrivateFunctionalities
			DoubleScalar updateRotationAngle(Scalar elapsedTime);
			#pragma endregion

			#pragma region ClassMembers
			/**Position, rotation, scale */
			VectorT m_position;
			VectorT m_eulerAngles;
			VectorT m_scale;
			VectorT m_direction;

			VectorT m_cameraPosition;

			/**Velocity */
			VectorT m_velocity;
			positionUpdate_t m_positionUpdate;
			rotationUpdate_t m_rotationUpdate;

			//Used for object window identification
			string m_objectName;
			#pragma endregion

			
		};

	}
	
}


#endif
