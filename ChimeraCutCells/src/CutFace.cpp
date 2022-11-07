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

#include "CutCells/CutFace.h"
//#include "CutCells/CutCells3D.h"

namespace Chimera {
	namespace CutCells {

		template<>
		void CutFace<Vector2>::updateCentroid() {
			Scalar signedArea = 0;
			for(int i = 0; i < m_cutEdges.size(); i++) {
				CutEdge<Vector2> *pEdge = m_cutEdges[i];
				Vector2 initialEdge = getEdgeInitialPoint(i);
				Vector2 finalEdge = getEdgeFinalPoint(i);
				Scalar currA = initialEdge.cross(finalEdge);
				signedArea += currA;
				m_centroid += (pEdge->getCentroid())*currA;
			}
			signedArea *= 0.5f;
			m_centroid /= (6.0f*signedArea);
		}

		template<>
		void CutFace<Vector3D>::updateCentroid() {
			Scalar signedAreaXY = 0, signedAreaXZ = 0, signedAreaYZ = 0;
			Vector2 centroidXY, centroidXZ, centroidYZ;

			
			for(int i = 0; i < m_cutEdges.size(); i++) {
				CutEdge<Vector3D> *pEdge = m_cutEdges[i];
				Vector3D initialEdge = getEdgeInitialPoint(i);
				Vector3D finalEdge = getEdgeFinalPoint(i);

				if (initialEdge == finalEdge)
					continue;

				//Projecting onto the XY plane
				Vector2 initialEdgeXY(initialEdge.x, initialEdge.y);
				Vector2 finalEdgeXY(finalEdge.x, finalEdge.y);

				Scalar currAreaXY = initialEdgeXY.cross(finalEdgeXY);
				signedAreaXY += currAreaXY;
				centroidXY.x += (pEdge->getCentroid().x)*currAreaXY;
				centroidXY.y += (pEdge->getCentroid().y)*currAreaXY;

				//Projecting onto the XZ plane
				Vector2 initialEdgeXZ(initialEdge.x, initialEdge.z);
				Vector2 finalEdgeXZ(finalEdge.x, finalEdge.z);

				Scalar currAreaXZ = initialEdgeXZ.cross(finalEdgeXZ);
				signedAreaXZ += currAreaXZ;
				centroidXZ.x += (pEdge->getCentroid().x)*currAreaXZ;
				centroidXZ.y += (pEdge->getCentroid().z)*currAreaXZ;

				//Projecting onto the YZ plane
				Vector2 initialEdgeYZ(initialEdge.y, initialEdge.z);
				Vector2 finalEdgeYZ(finalEdge.y, finalEdge.z);

				Scalar currAreaYZ = initialEdgeYZ.cross(finalEdgeYZ);
				signedAreaYZ += currAreaYZ;
				centroidYZ.x += (pEdge->getCentroid().y)*currAreaYZ;
				centroidYZ.y += (pEdge->getCentroid().z)*currAreaYZ;
			}
			
			signedAreaXZ *= 0.5f;
			signedAreaXY *= 0.5f;
			signedAreaYZ *= 0.5f;
			
			if (abs(signedAreaXZ) < 1e-4  && abs(signedAreaXY) < 1e-4) {
				m_centroid.x = getEdgeInitialPoint(0).x;
			} else if (abs(signedAreaXZ) > abs(signedAreaXY)) {
				m_centroid.x = centroidXZ.x / (3.0f*signedAreaXZ);
			}
			else {
				m_centroid.x = centroidXY.x / (3.0f*signedAreaXY);
			}

			if (abs(signedAreaXY) < 1e-4  && abs(signedAreaYZ) < 1e-4) {
				m_centroid.y = getEdgeInitialPoint(0).y;
			} else  if (abs(signedAreaXY) > abs(signedAreaYZ)) {
				m_centroid.y = centroidXY.y / (3.0f*signedAreaXY);
			}
			else {
				m_centroid.y = centroidYZ.x / (3.0f*signedAreaYZ);
			}
			
			if (abs(signedAreaXZ) < 1e-4  && abs(signedAreaYZ) < 1e-4) {
				m_centroid.z = getEdgeInitialPoint(0).z;
			} else if (abs(signedAreaXZ) > abs(signedAreaYZ)) {
				m_centroid.z = centroidXZ.y / (3.0f*signedAreaXZ);
			}
			else {
				m_centroid.z = centroidYZ.y / (3.0f*signedAreaYZ);
			}

			//return m_centroid;
		}

		template<>
		bool CutFace<Vector2>::intersectsGeometryEdges(const Vector2 &p1, const Vector2 &p2) const {
			for (int i = 0; i < m_cutEdges.size(); i++) {
				if (m_cutEdgesLocations[i] == geometryEdge) {
					if (DoLinesIntersect(p1, p2, getEdgeInitialPoint(i), getEdgeFinalPoint(i)))
						return true;
				}
			}
			return false;
		}

		template<>
		bool CutFace<Vector3>::intersectsGeometryEdges(const Vector3 &p1, const Vector3 &p2) const {
			return false;
		}

		template<>
		bool CutFace<Vector3D>::intersectsGeometryEdges(const Vector3D &p1, const Vector3D &p2) const {
			return false;
		}
	}
}