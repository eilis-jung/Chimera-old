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

#pragma once
#ifndef __CHIMERA_FACE_H_
#define __CHIMERA_FACE_H_

#include "ChimeraCore.h"
#include "MeshesCoreDefs.h"
#include "Mesh/Edge.h"

namespace Chimera {

	namespace Meshes {

		template <class VectorType>
		class Face;
		/** Since there's no "T-faces", we can store halfFaceLocation on halfFace class. This is not possible for 
			half-edges.*/
		template <class VectorType>
		class HalfFace {
		public:
			#pragma region Constructors
			/**Effortless constructor: all half-edges are already precomputed */
			HalfFace(const vector<HalfEdge<VectorType> *> &halfEdges, Face<VectorType> *pFace, halfFaceLocation_t halfFaceLocation = backHalfFace) : m_halfEdges(halfEdges), m_streamfunctions(halfEdges.size()) {
				m_pFace = pFace;
				m_ID = m_currID++;
				m_faceLocation = halfFaceLocation;
				m_centroid = computeCentroid();
				m_normal = computeNormal();
			}

			/** This constructor creates its internal half-edges by acessing vertices on a vector by indices passed 
				on a ordered vertices indices vector (adjacent indices are continuous edges. It has no parent face, 
				since its used on polygonal meshes */
			HalfFace(const vector<Vertex<VectorType> *> &vertices, vector<uint> vertexIndices, Face<VectorType> *pFace = NULL, halfFaceLocation_t halfFaceLocation = geometryHalfFace) {
				for (int i = 0; i < vertexIndices.size(); i++) {
					int nextI = roundClamp<int>(i + 1, 0, vertexIndices.size());
					Edge<VectorType> *pEdge = new Edge<VectorType>(vertices[vertexIndices[i]], vertices[vertexIndices[nextI]], geometricEdge);
					m_halfEdges.push_back(new HalfEdge<VectorType>(vertices[vertexIndices[i]], vertices[vertexIndices[nextI]], pEdge));
					m_halfEdges.back()->setLocation(geometryHalfEdge);
				}
				m_streamfunctions.resize(m_halfEdges.size());
				m_pFace = pFace;
				m_ID = m_currID++;
				m_faceLocation = geometryHalfFace;

				m_normal = computeNormal();
				m_centroid = computeCentroid();
			}

			/** Special constructor: creates a pair of half-faces that have opposite half-edges */
			static pair<HalfFace<VectorType> *, HalfFace<VectorType>*> createDualFaces(const vector<Vertex<VectorType> *> &vertices, vector<uint> vertexIndices, Face<VectorType> *pFace = NULL, halfFaceLocation_t halfFaceLocation = geometryHalfFace) {
				HalfFace<VectorType> *pFirstFace = new HalfFace<VectorType>(vertices, vertexIndices, pFace, halfFaceLocation);
				vector<HalfEdge<VectorType> *> dualHalfEdges;
				uint hfSize = pFirstFace->getHalfEdges().size();
				dualHalfEdges.resize(hfSize);
				
				/** Creating half-edges with reversed index inside the vector and reversed vertices */
				for (int i = 0; i < dualHalfEdges.size(); i++) {
					uint revIndex = hfSize - (i + 1);
					HalfEdge<VectorType> *pHalfEdge = pFirstFace->getHalfEdges()[revIndex];
					dualHalfEdges[i] = new HalfEdge<VectorType>(pHalfEdge->getVertices().second,
																pHalfEdge->getVertices().first, 
																pHalfEdge->getEdge());
					dualHalfEdges[i]->setLocation(geometryHalfEdge);
				}
				
				HalfFace<VectorType> *pDualFace = new HalfFace<VectorType>(dualHalfEdges, pFace, halfFaceLocation);
				return pair<HalfFace<VectorType> *, HalfFace<VectorType> *>(pFirstFace, pDualFace);
			}

			static void resetIDs() {
				m_currID = 0;
			}
			#pragma endregion 

			#pragma region AccessFunctions
			const vector<HalfEdge<VectorType> *> & getHalfEdges() const {
				return m_halfEdges;
			}

			vector<HalfEdge<VectorType> *> & getHalfEdges() {
				return m_halfEdges;
			}

			uint getID() const {
				return m_ID;
			}

			halfFaceLocation_t getLocation() const {
				return m_faceLocation;
			}

			void setLocation(halfFaceLocation_t halfFaceLocation) {
				m_faceLocation = halfFaceLocation;
			}

			Face<VectorType> * getFace() {
				return m_pFace;
			}

			const Face<VectorType> * getFace() const {
				return m_pFace;
			}

			void setFace(Face<VectorType> *pFace) {
				m_pFace = pFace;
			}

			const VectorType & getStreamfunction(uint i) const {
				return m_streamfunctions[i];
			}

			void setStreamfunction(const VectorType &streamfunction, uint i) {
				m_streamfunctions[i] = streamfunction;
			}

			const VectorType & getCentroid() const {
				return m_centroid;
			}

			const VectorType & getNormal() const {
				return m_normal;
			}

			void setNormal(const VectorType &normal) {
				m_normal = normal;
			}

			#pragma endregion

			#pragma region Operators
			FORCE_INLINE bool operator==(const HalfFace<VectorType> &rhs) const {
				return m_ID == rhs.getID();
			}
			#pragma endregion
			
			#pragma region Functionalities
			/** Gets the adjacent half-face connected to this face through an edge if its on the same side of the geometry.*/
			HalfFace<VectorType> * getSameSideHalfFaceConnectedToEdge(uint edgeID) {
				Edge<VectorType> *pCurrEdge = m_halfEdges[edgeID]->getEdge();
				if (pCurrEdge->getConnectedHalfFaces().size() == 1 /*|| m_halfEdges[edgeID]->getLocation() == geometryHalfEdge*/) {
					//Connected to this face only, return null
					return nullptr;
				}
				if (pCurrEdge->getConnectedHalfFaces()[0]->getID() == m_ID) {	
					return pCurrEdge->getConnectedHalfFaces()[1];
				}
				return pCurrEdge->getConnectedHalfFaces()[0];
			}

			/** Verifies is a line segment crossed through geometry */
			bool crossedThroughGeometry(const VectorType &v1, const VectorType &v2, VectorType &crossingPoint) const;

			/** Calculates the smallest distance of a point to geometric edges*/
			Scalar getDistanceToBoundary(const VectorType & position) const;

			/** Checks if a given point is inside this halfFace*/
			bool isInside(const VectorType &position) const;

			/** For a geometric half-face (triangle), checks if a ray intersects it */
			bool rayIntersect(const VectorType &point, const VectorType &rayDirection);

			/** Checks if this half-face has a certain half-edge */
			bool hasHalfedge(HalfEdge<VectorType> *pHalfEdge);

			/** Checks if this half-face has a half-edge, but in a reversed direction*/
			bool hasReverseHalfedge(HalfEdge<VectorType> *pHalfEdge);

			/** Checks if this half-face has a certain edge */
			bool hasEdge(Edge<VectorType> *pEdge);

			/** Creates half-face with a opposite orientation from this one. Useful for 3-D faces routine initialization */
			HalfFace<VectorType> * reversedCopy();

			/** Does a reverse copy, but replaces the internal half-edge vertices by the vertices on the map. Useful
				for ghost vertices initialization*/
			HalfFace<VectorType> * reversedCopy(const map<uint, Vertex<VectorType> *> &ghostVerticesMap);
			#pragma endregion 

		protected:
			#pragma region ClassMembers
			/** Halfedges, father face, and face location in 3-D */
			vector<HalfEdge<VectorType> *> m_halfEdges;
			Face<VectorType> *m_pFace;
			halfFaceLocation_t m_faceLocation;

			/** Vector of ordered vertices represented by this halfface. This accelerator structure is widely useful
				when interpolating values inside a mesh. */
			vector<Vertex<VectorType>*> m_vertices;

			/** Since streamfunctions are discontinuous per cell, we are storing them on half-faces*/
			vector<VectorType> m_streamfunctions;

			/** Face vector attributes */
			VectorType m_centroid;
			VectorType m_normal;

			/** ID vars */
			uint m_ID;
			static uint m_currID;
			#pragma endregion 

			#pragma region PrivateFunctionalities
			VectorType computeCentroid();

			VectorType computeNormal();
			#pragma endregion 

		};


		template <class VectorType>
		class HalfVolume;

		/** Face structure: stores centroids and attributes relative to faces. It has the ownership of two half-faces,
		* which are created on the construction of this class. This might have a non-manifold structure, with T-vertices. */
		template <class VectorType>
		class Face {
		public:

			#pragma region Constructors
			Face(const vector<Edge<VectorType> *> &edges, dimensions_t cellLocation, DoubleScalar gridDx, faceLocation_t faceLocation = XYFace) : m_edges(edges), m_vertexToEdgeMap(edges) {
				m_ID = m_currID++;
				m_gridCellLocation = cellLocation;
				m_gridDx = gridDx;
				m_visited = true;
				m_faceLocation = faceLocation;
			}

			/**Empty face constructor */
			Face(faceLocation_t faceLocation = XYFace) {
				m_ID = m_currID++;
				m_visited = false;
				m_faceLocation = faceLocation;
			}

			static void resetIDs() {
				m_currID = 0;
			}
			#pragma endregion 

			#pragma region Functionalities
			/** Splits this face into a series of half-faces. It uses simples 2-D heuristics to make correct turns. *
				Initializes the internal faces vector structure. Stores the result in m_halfFaces structure */
			const vector<HalfFace<VectorType> *> & split();
			
			/** Looks for the halfface that this point is contained */
			HalfFace<VectorType> * getHalfFace(const VectorType &position);

			VectorType computeFaceNormal();

			/** Only works for 3-D, where this face half-faces are mirrored */
			DoubleScalar calculateArea();

			/** Convert all the half-faces of this face to faces. Each face is created with half-faces that are mirrors 
				of each other - like half-edges, but in 3-D. */
			vector<Face<VectorType> *> convertToFaces3D();
			#pragma endregion


			#pragma region AccessFunctions
			uint getID() const {
				return m_ID;
			}

			const VectorType & getCentroid() const {
				return m_centroid;
			}

			void setCentroid(const VectorType &centroid) {
				m_centroid = centroid;
			}

			const VectorType & getVelocity() const {
				return m_velocity;
			}

			void setVelocity(const VectorType &velocity) {
				m_velocity = velocity;
			}

			const VectorType & getAuxiliaryVelocity() const {
				return m_auxVelocity;
			}

			void setAuxiliaryVelocity(const VectorType &auxVelocity) {
				m_auxVelocity = auxVelocity;
			}


			faceLocation_t getLocation() const {
				return m_faceLocation;
			}

			const vector<Edge<VectorType> *> & getEdges() const {
				return m_edges;
			}

			const VertexToEdgeMap<VectorType> & getVertexToEdgeMap() const {
				return m_vertexToEdgeMap;
			}

			VertexToEdgeMap<VectorType> & getVertexToEdgeMap() {
				return m_vertexToEdgeMap;
			}

			void setGridCellLocation(const dimensions_t &gridCellLocation) {
				m_gridCellLocation = gridCellLocation;
			}
			const dimensions_t & getGridCellLocation() const {
				return m_gridCellLocation;
			}

			const vector<HalfFace<VectorType> *> & getHalfFaces() {
				return m_halfFaces;
			}

			/** Useful for setting half-faces that have opposing half-edges on cut-cells */
			void swapHalfFaces() {
				swap(m_halfFaces[0], m_halfFaces[1]);
			}

			void addHalfFace(HalfFace<VectorType> *pHalfFace) {
				m_halfFaces.push_back(pHalfFace);
			}

			bool isVisited() const {
				return m_visited;
			}

			void setVisited(bool visited) {
				m_visited = visited;
			}

			/** Appends an face that is connected to this edge. Does not verify duplicates for performance */
			void addConnectedHalfVolume(HalfVolume<VectorType> *pVolume) {
				m_connectedHalfVolumes.push_back(pVolume);
			}

			const vector<HalfVolume<VectorType> *> & getConnectedHalfVolumes() {
				return m_connectedHalfVolumes;
			}


			void setRelativeFraction(DoubleScalar fraction) {
				m_areaFraction = fraction;
			}

			DoubleScalar getRelativeFraction() const {
				return m_areaFraction;
			}

			/** This function is called on the special case that this face is initialized with halfedges (poly3D ctor).
				Then it initializes its internal edges vector with the halfedges from the first face. The second face
				is just a mirrored copy of the first one - and this face should only have two half-faces.*/
			void initializeEdgesFromHalfedges() {
				if (m_halfFaces.size() != 2) {
					throw(exception("Face initializeEdgesFromHalfedges: incorrect number of initial half-faces"));
				}
				for (int i = 0; i < m_halfFaces.front()->getHalfEdges().size(); i++) {
					m_edges.push_back(m_halfFaces.front()->getHalfEdges()[i]->getEdge());
				}
			}

			#pragma endregion

			#pragma region Operators
			FORCE_INLINE bool operator==(const Face<VectorType> &rhs) const {
				return m_ID == rhs.getID();
			}
			#pragma endregion
		protected:
			#pragma region ClassMembers
			/**Undirected edges of this face */
			vector<Edge<VectorType> *> m_edges;

			/** Half faces */
			vector<HalfFace<VectorType> *> m_halfFaces;

			/** Velocity values stored per face: velocity is usually the div-free velocity field after projection. 
			  * Aux velocity is the unprojected velocity after advection */
			VectorType m_velocity;
			VectorType m_auxVelocity;

			/** Regular grid location and scale, if applicable */
			dimensions_t m_gridCellLocation;
			DoubleScalar m_gridDx;

			/** Each face has a vertex to edge map that maps the elements of that specific face. This is a helper structure
				that selectes the edges connected to the vertex of this specific face. This is usefeul when splitting the 
				face into halfFaces. */
			VertexToEdgeMap<VectorType> m_vertexToEdgeMap;

			/* Half volumes connected to this face */
			vector<HalfVolume<VectorType> *> m_connectedHalfVolumes;

			/** 3-D plane face location, if applicable */
			faceLocation_t m_faceLocation;

			/** Centroid, if applicable */
			VectorType m_centroid;

			/** Relative area fraction */
			DoubleScalar m_areaFraction;

			/** ID vars */
			uint m_ID;
			static uint m_currID;

			/** Helper for 3-D cut-cells subdivision */
			bool m_visited;

			#pragma endregion

			#pragma region PrivateFunctionalities
			/** Search for unvisited edges. EdgeID will contain the first non-visited element found. */
			Edge<VectorType> * hasUnvisitedEdges() {
				for (uint i = 0; i < m_edges.size(); i++) {
					if (m_edges[i]->getType() != geometricEdge && !m_edges[i]->isVisited()) {
						return m_edges[i];
					} /*else if(m_edges[i].getEdgeType == geometricEdge && 
								(!m_edges[i]->getHalfEdges().first->isVisited() || 
								  m_edges[i]->getHalfEdges().second->isVisited())  {
						edgeID = i;

					}*/
				}
				return nullptr;
			}

			/** Depth first seach: in 2-D the connectivity graph does not branches, but the depth-first principle is 
				applied. This recursive function will call itself until it has visited all edges on a cycle. The visited
				edges will be appended to halfedges vector. */
			void breadthFirstSearch(Edge<VectorType> *pEdge, vector<HalfEdge<VectorType> *> &halfEdges);

			/** Halfface location trick for making the correct turn when coming from a geometry edge into a grid edge.
			    Uses the idea that a face should always be counterclockwise to find the correct half-edge given a location.
				Also considers that the first halfedge of edge will be positive vector relative to the Cartesian space. */
			HalfEdge<VectorType> * getOrientedHalfEdge(Edge<VectorType> *pEdge, halfEdgeLocation_t halfEdgeLocation);
			#pragma endregion
		};

		template <class VectorType>
		unsigned int HalfFace<VectorType>::m_currID = 0;

		template <class VectorType>
		unsigned int Face<VectorType>::m_currID = 0;

		#pragma region HelperStructures
		/** Given an edge ID, gets all the faces that are connected to this edge */
		template <class VectorType>
		class EdgeToFaceMap {

		public:
			//Empty constructor
			EdgeToFaceMap() { };
			EdgeToFaceMap(const vector<Face<VectorType> *> &faces) {
				initializeMap(faces);
			};

			const vector<Face<VectorType> *> & getFaces(Edge<VectorType> *pEdge) {
				return m_edgeToFaceMap[pEdge->getID()];
			}

			void initializeMap(const vector<Face<VectorType> *> &faces);

		protected:
			map<uint, vector<Face<VectorType> *>> m_edgeToFaceMap;
		};
		#pragma endregion
	}
}

#endif