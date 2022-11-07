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
//	

#include "Rigid/RigidObject2D.h"
#include "RenderingUtils.h"
#include <iomanip>      // std::setprecision

namespace Chimera {


	namespace Solids {

		template <class VectorT>
		RigidObject2D<VectorT>::RigidObject2D(LineMesh<VectorT>* pLineMesh, positionUpdate_t positionUpdate, rotationUpdate_t rotationUpdate, couplingType_t couplingType) :
			PhysicalObject(VectorT(0, 0), VectorT(0, 0), VectorT(1, 1)) {
			m_ID = m_currID++;
			m_pLineMesh = pLineMesh;
			m_positionUpdate = positionUpdate;
			m_rotationUpdate = rotationUpdate;
			m_couplingType = couplingType;
			m_pLineMesh->getParams()->updateCentroid();
			if (abs(m_rotationUpdate.initialRotation) > 0) {
				//Translate according centroid to rotate
				for (int i = 0; i < m_pLineMesh->getParams()->initialPoints.size(); i++) {
					m_pLineMesh->getParams()->initialPoints[i] -= m_pLineMesh->getParams()->centroid;
				}

				//Rotate
				for (int i = 0; i < m_pLineMesh->getParams()->initialPoints.size(); i++) {
					m_pLineMesh->getParams()->initialPoints[i].rotate(m_rotationUpdate.initialRotation);
				}

				//Translate according centroid to rotate
				for (int i = 0; i < m_pLineMesh->getParams()->initialPoints.size(); i++) {
					m_pLineMesh->getParams()->initialPoints[i] += m_pLineMesh->getParams()->centroid;
				}
				m_rotationUpdate.initialRotation = 0;
			}
			m_pLineMesh->updatePoints();
			m_pLineMesh->setHasUpdated(false);
			m_initialParams = m_updatedParams = *m_pLineMesh->getParams();
			
		}

		
		template <class VectorT>
		void RigidObject2D<VectorT>::update(Scalar dt) {
			Scalar currTime = PhysicsCore<VectorT>::getInstance()->getElapsedTime() + dt;
		
			m_pLineMesh->setHasUpdated(false);
			
			if (m_rotationUpdate.rotationType != noRotation) {
				//Rotate first in the initial frame of reference
				//Translate according centroid to rotate
				m_updatedParams.initialPoints = m_initialParams.initialPoints;
				for (int i = 0; i < m_updatedParams.initialPoints.size(); i++) {
					m_updatedParams.initialPoints[i] -= m_initialParams.centroid;
				}

				//Rotate
				Scalar rotationAngle = m_rotationUpdate.update2D(currTime);
				for (int i = 0; i < m_updatedParams.initialPoints.size(); i++) {
					m_updatedParams.initialPoints[i].rotate(rotationAngle);
				}
				//Translate according centroid to rotate
				for (int i = 0; i < m_updatedParams.initialPoints.size(); i++) {
					m_updatedParams.initialPoints[i] += m_updatedParams.centroid;
				}

				if(rotationAngle != 0)
					m_pLineMesh->setHasUpdated(true);
			}

			if (m_positionUpdate.positionUpdateType != noPositionUpdate) {
				if (m_rotationUpdate.rotationType == noRotation) {
					m_updatedParams.initialPoints = m_initialParams.initialPoints;
				}
				//Then update position
				for (int i = 0; i < m_updatedParams.initialPoints.size(); i++) {
					if (currTime > m_positionUpdate.startingTime && currTime < m_positionUpdate.endingTime) {
						m_updatedParams.initialPoints[i] += m_positionUpdate.update(currTime);
						m_pLineMesh->setHasUpdated(true);
					}		
				}
			}
			if(m_pLineMesh->hasUpdated())
				m_pLineMesh->updatePoints(m_updatedParams);
		}

		template <class VectorT>
		void RigidObject2D<VectorT>::updateCutEdgesVelocities(int timeOffset, Scalar dx, bool useAuxiliaryVelocities) {
			if (m_positionUpdate.positionUpdateType == noPositionUpdate && m_rotationUpdate.rotationType == noRotation) {
				return;
			}

			Scalar dt = PhysicsCore<VectorT>::getInstance()->getParams()->timestep;

			Scalar currTime = PhysicsCore<VectorT>::getInstance()->getElapsedTime() + dt*(timeOffset + 1);
			//Use temp params to predict next position
			LineMesh<VectorT>::params_t tempParams(m_initialParams);
			//Translate according centroid to rotate
			for (int i = 0; i < tempParams.initialPoints.size(); i++) {
				tempParams.initialPoints[i] -= tempParams.centroid;
			}
			//Rotate
			for (int i = 0; i < tempParams.initialPoints.size(); i++) {
				tempParams.initialPoints[i].rotate(m_rotationUpdate.update2D(currTime));
			}
			//Translate according centroid to reverse initial translation
			for (int i = 0; i < tempParams.initialPoints.size(); i++) {
				tempParams.initialPoints[i] += tempParams.centroid;
			}
			
			//Add the position update
			for (int i = 0; i < tempParams.initialPoints.size(); i++) {
				if (currTime > m_positionUpdate.startingTime && currTime < m_positionUpdate.endingTime) {
					tempParams.initialPoints[i] += m_positionUpdate.update(currTime);
				}
			}

			//Then calculate the velocities on geometry vertices
			int ti = 0;
			for (int i = 0; i < m_pLineMesh->getElements().size(); i++) {
				if (m_pLineMesh->getElements()[i]->getVertex1()->getVertexType() == geometryVertex) {
					//Because of ghost vertices, we have to access vertices through halfedges
					Vertex<VectorT> *pV1 = m_pLineMesh->getElements()[i]->getVertex1();
					Vertex<VectorT> *pV2 = m_pLineMesh->getElements()[i]->getHalfEdges().second->getVertices().second;
					if (pV1->getPosition() != pV2->getPosition()) {
						throw("Error on ghost vertex velocity update");
					}
					VectorT nextPos = tempParams.initialPoints[ti] - tempParams.centroid;
					VectorT prevPos = m_updatedParams.initialPoints[ti] - m_updatedParams.centroid;
					VectorT effectiveVelocity = (nextPos - prevPos)/dt;
					if(useAuxiliaryVelocities) {
						pV1->setAuxiliaryVelocity(effectiveVelocity);
						pV2->setAuxiliaryVelocity(effectiveVelocity);
					}
					else {
						pV1->setVelocity(effectiveVelocity);
						pV2->setVelocity(effectiveVelocity);
					}
					ti++;
				}
			}
			if (ti != tempParams.initialPoints.size() - 1 && ti != tempParams.initialPoints.size()) {
				throw("Calculating vertices effective velocities GONE WRONG");
			}

			for (int i = 0; i < m_pLineMesh->getElements().size(); i++) {
				if (m_pLineMesh->getElements()[i]->getVertex1()->getVertexType() == geometryVertex) {
					//Find next geometry vertex and total size first
					Vertex<VectorT> *pInitialVertex = m_pLineMesh->getElements()[i]->getVertex1();
					Vertex<VectorT> *pNextVertex = nullptr;
					DoubleScalar totalSize = 0;
					int j;
					for (j = roundClamp<int>(i + 1, 0, m_pLineMesh->getElements().size()); /**Only stops if find the next vertex*/; j = roundClamp<int>(j + 1, 0, m_pLineMesh->getElements().size())) {
						int prevJ = roundClamp<int>(j - 1, 0, m_pLineMesh->getElements().size());
						totalSize += (m_pLineMesh->getElements()[j]->getVertex1()->getPosition() - m_pLineMesh->getElements()[prevJ]->getVertex1()->getPosition()).length();
						if (m_pLineMesh->getElements()[j]->getVertex1()->getVertexType() == geometryVertex) {
							pNextVertex = m_pLineMesh->getElements()[j]->getVertex1();
							break;
						}
					}
					if (i + 1 == j) {
						continue;
					}

					//Cycle through non geometry vertices and initialize their velocities as linear combinations of nearby geometry edges
					DoubleScalar alpha = 0;
					j = i + 1;
					for (; j < m_pLineMesh->getElements().size(); j++) {
						if (m_pLineMesh->getElements()[j]->getVertex1()->getVertexType() == geometryVertex) {
							break;
						}
						alpha += (m_pLineMesh->getElements()[j]->getVertex1()->getPosition() - m_pLineMesh->getElements()[j - 1]->getVertex1()->getPosition()).length()/totalSize;
						VectorT vertexVelocity;
						if(useAuxiliaryVelocities)
							vertexVelocity = pInitialVertex->getAuxiliaryVelocity()*(1 - alpha) + pNextVertex->getAuxiliaryVelocity()*alpha;
						else
							vertexVelocity = pInitialVertex->getVelocity()*(1 - alpha) + pNextVertex->getVelocity()*alpha;

						Vertex<VectorT> *pV1 = m_pLineMesh->getElements()[j]->getVertex1();
						Vertex<VectorT> *pV2 = m_pLineMesh->getElements()[j]->getHalfEdges().second->getVertices().second;
						if (useAuxiliaryVelocities) {
							pV1->setAuxiliaryVelocity(vertexVelocity);
							pV2->setAuxiliaryVelocity(vertexVelocity);
						}
						else {
							pV1->setVelocity(vertexVelocity);
							pV2->setVelocity(vertexVelocity);
						}
						
					}
				}
			}

			for (int i = 0; i < m_pLineMesh->getElements().size(); i++) {
				Vertex<VectorT> *pV1 = m_pLineMesh->getElements()[i]->getVertex1();
				Vertex<VectorT> *pV2 = m_pLineMesh->getElements()[i]->getVertex2();
				if(useAuxiliaryVelocities)
					m_pLineMesh->getElements()[i]->setAuxiliaryVelocity((pV1->getAuxiliaryVelocity() + pV2->getAuxiliaryVelocity())*0.5);
				else
					m_pLineMesh->getElements()[i]->setVelocity((pV1->getVelocity() + pV2->getVelocity())*0.5);
			}
		}

		template <class VectorT>
		unsigned int RigidObject2D<VectorT>::m_currID = 0;

		template class RigidObject2D<Vector2>;
	}

	
}