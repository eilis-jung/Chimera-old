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

#ifndef __CHIMERA_VERTEX_H_
#define __CHIMERA_VERTEX_H_

#include "ChimeraCore.h"
#include "MeshesCoreDefs.h"
#include "CGALConfig.h"

namespace Chimera {
	using namespace Core;

	namespace Meshes {
		

		/** Forward edge/faces declaration */
		template <class VectorType>
		class Edge;

		template <class VectorType>
		class Face;

		template <class VectorType>
		class HalfFace;

		/** Vertex structure: stores the location of the vertex, which is paramount for ensuring correct cut-cells 
			creation. It does owns this vertex, so updates have to be done on the vertex itself. */
		template <class VectorType>
		class Vertex {
		public:

			#pragma region Constructors

			/** Just here for compilation compatibility issues: not actually used */
			Vertex() {
				m_vertexID = UINT_MAX;
			}

			Vertex(const VectorType &vertexPosition, vertexType_t vertexType, CGALWrapper::CgalPolyhedron::Vertex *pCGALVertex = nullptr) : m_position(vertexPosition) {
				m_vertexID = m_currID++;
				m_vertexType = vertexType;
				m_nodeWeight = 0.0;
				m_nodePressureGradient = VectorType();
				m_hasUpdated = false;
				m_onGridNode = false;
				m_pCGALVertex = pCGALVertex;
			}

			static void resetIDs() {
				m_currID = 0;
			}
			#pragma endregion

			#pragma region AccessFunctions
			const VectorType & getPosition() const {
				return m_position;
			}

			void setPosition(const VectorType &position) {
				m_position = position;
			}

			const VectorType & getNormal() const {
				return m_normal;
			}
			void setNormal(const VectorType &normal) {
				m_normal = normal;
			}

			uint getID() const {
				return m_vertexID;
			}

			vertexType_t getVertexType() const {
				return m_vertexType;
			}

			void setVertexType(vertexType_t vertexType) {
				m_vertexType = vertexType;
			}

			/** Node properties access functions */
			void setVelocity(const VectorType &velocity) {
				m_velocity = velocity;
			}

			const VectorType & getVelocity() const {
				return m_velocity;
			}

			void addAuxiliaryVelocity(const VectorType &auxVelocity){
				m_auxiliaryVelocity += auxVelocity;
			}

			void setAuxiliaryVelocity(const VectorType &auxVelocity) {
				m_auxiliaryVelocity = auxVelocity;
			}

			const VectorType & getAuxiliaryVelocity() const {
				return m_auxiliaryVelocity;
			}

			void setWeight(DoubleScalar nodeWeight) {
				m_nodeWeight = nodeWeight;
			}

			DoubleScalar getWeight()  const {
				return m_nodeWeight;
			}

			void addWeight(DoubleScalar nodeWeight) {
				m_nodeWeight += nodeWeight;
			}

			void setPressureGradient(VectorType nodePressureGradient) {
				m_nodePressureGradient = nodePressureGradient;
			}

			VectorType getPressureGradient()  const {
				return m_nodePressureGradient;
			}

			void addPressureGradient(VectorType nodePressureGradient) {
				m_nodePressureGradient += nodePressureGradient;
			}

			void setOnGridNode(bool onGridNode) {
				m_onGridNode = onGridNode;
			}

			bool isOnGridNode() const {
				return m_onGridNode;
			}

			/** Node connectivity access functions */
			/** Appends an edge that is connected to this vertex. Does not verify duplicates for performance */
			void addConnectedEdge(Edge<VectorType> *pEdge) {
				for (int i = 0; i < m_connectedEdges.size(); i++) {
					if (m_connectedEdges[i] == pEdge)
						return;
					if (m_connectedEdges[i]->getHalfEdges().first == pEdge->getHalfEdges().first ||
						m_connectedEdges[i]->getHalfEdges().first == pEdge->getHalfEdges().second ||
						m_connectedEdges[i]->getHalfEdges().second == pEdge->getHalfEdges().first ||
						m_connectedEdges[i]->getHalfEdges().second == pEdge->getHalfEdges().second)
						return;
				}
				m_connectedEdges.push_back(pEdge);
			}
			
			const vector<Edge<VectorType> *> & getConnectedEdges() {
				return m_connectedEdges;
			}

			void clearConnectedEdges() {
				m_connectedEdges.clear();
			}

			/** Appends an face that is connected to this vertex. Does not verify duplicates for performance */
			void addConnectedFace(Face<VectorType> *pFace) {
				m_connectedFaces.push_back(pFace);
			}

			const vector<Face<VectorType> *> getConnectedFaces() {
				return m_connectedFaces;
			}

			void clearConnectedFaces() {
				m_connectedFaces.clear();
			}

			/** Appends a half face that is connected to this vertex. Does not verify duplicates for performance */
			void addConnectedHalfFace(HalfFace<VectorType> *pHalfFace) {
				m_connectedHalfFaces.push_back(pHalfFace);
			}

			const vector<HalfFace<VectorType> *> getConnectedHalfFaces() {
				return m_connectedHalfFaces;
			}
			
			void clearConnectedHalfFaces() {
				m_connectedHalfFaces.clear();
			}

			/** Update utility*/
			bool hasUpdated() const {
				return m_hasUpdated;
			}

			void setUpdated(bool updated) {
				m_hasUpdated = updated;
			}

			bool isOnGeometryVertex() {
				if (m_vertexType == geometryVertex || m_vertexType == edgeVertex || m_vertexType == faceVertex)
					return true;
				return false;
			}

			#pragma endregion

			#pragma region Operators
			FORCE_INLINE bool operator==(const Vertex<VectorType> &rhs) const {
				return m_vertexID == rhs.getID();
			}
			#pragma endregion
		protected:
			#pragma region ClassMembers
			VectorType m_position;
			vertexType_t m_vertexType;

			/** Common vertices attributes */
			VectorType m_velocity;
			VectorType m_normal;
			VectorType m_vorticity;
			VectorType m_auxiliaryVelocity;

			/** Advection auxiliary variables */
			DoubleScalar m_nodeWeight;

			/** Advection auxiliary variables */
			VectorType m_nodePressureGradient;

			/** Edges connected to this vertex */
			vector<Edge<VectorType> *> m_connectedEdges;

			/** Faces connected to this vertex */
			vector<Face<VectorType> *> m_connectedFaces;

			/** Half-faces connected to this vertex */
			vector<HalfFace<VectorType>*> m_connectedHalfFaces;

			/** Utility used for mixed node velocity computation */
			bool m_hasUpdated;

			/** Special flag used if a vertex is on top of a grid node */
			bool m_onGridNode;

			/** Possible Handle to CGALPolyhedron::Vertex */
			CGALWrapper::CgalPolyhedron::Vertex * m_pCGALVertex;

			/** ID vars */
			uint m_vertexID;
			static uint m_currID;

			#pragma endregion
		};

		template <class VectorType>
		unsigned int Vertex<VectorType>::m_currID = 0;
	}
}

#endif