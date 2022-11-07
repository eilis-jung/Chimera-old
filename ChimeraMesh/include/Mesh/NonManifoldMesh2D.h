
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

#ifndef _CHIMERA_DATA_NON_MANIFOLD_MESH_2D_H_
#define _CHIMERA_DATA_NON_MANIFOLD_MESH_2D_H_

#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "Mesh/LineMesh.h"
#include "Mesh/Mesh.h"

namespace Chimera {
	
	using namespace Core;
	using namespace Grids;
	using namespace CutCells;

	namespace CutCells {
		template <class VectorT>
		class CutFace;

		//In C++0x is valid to forward declare enums 
		enum edgeLocation_t : int;
		enum faceLocation_t : int;
	}

	namespace Meshes {


		class NonManifoldMesh2D {
		public:

			#pragma region InternalStructures
			typedef struct edge_t {
				unsigned int indices[2];
				unsigned int ID;
				unsigned int degree;

				CutEdge<Vector3D> *pCutEdge;
				//Valid Interior point used for isLeft comparisons
				Vector3D centroid;

				edgeLocation_t edgeLocation;
				bool crossesGrid;

				bool addedInverted;

				edge_t(unsigned int g_ID) {
					degree = 0;
					ID = g_ID;
					pCutEdge = NULL;
					crossesGrid = true;
					addedInverted = false;
				}
			} edge_t;


			class DisconnectedRegion {
				Vector3D m_centroid;
				vector<Vector3D> m_points;
				bool m_closedRegion;

			public:
				DisconnectedRegion(const Meshes::lineSegment_t<Vector3D> &lineSegment, faceLocation_t faceLocation);

				const Vector3D & getCentroid() const {
					return m_centroid;
				}

				int getNumberOfPoints() const {
					return m_points.size();
				}

				const Vector3D & getPoint(int index) const {
					return m_points[index];
				}

				//Returns the closest point from this disconnected region relative to the point
				Vector3D getClosestPoint(const Vector3D &point) const;
				//Returns the closes points from both Disconnected regions
				pair<Vector3D, Vector3D> getClosestPoint(const DisconnectedRegion &disconnectedRegion) const;
			};

			#pragma endregion

			#pragma region Constructors
			NonManifoldMesh2D(dimensions_t cellDim, faceLocation_t faceSliceLocation, const vector<lineSegment_t<Vector3D>> &lineSegments, Scalar dx);
			#pragma endregion

			#pragma region Functionalities
			vector<CutFace<Vector3D> *> split();
			#pragma endregion

			#pragma region AcessFunctions
			const vector<edge_t> & getEdges() const {
				return m_edges;
			}
			const dimensions_t & getFaceDimensions() const {
				return m_faceDim;
			}
			#pragma endregion


		private:
			#pragma region PrivateMembers
			/** Edges, vertices and cut-cells reference */
			vector<Vector3D> m_vertices;
			vector<edge_t> m_edges;
			vector<bool> m_visitedEdges;
			vector<int> m_visitedEdgesCount;
			Scalar m_dx;
			vector<lineSegment_t<Vector3D>> m_lineSegments;
			/** Equivalent to openEndedCell, but for each line segment. Used in signedDistance tag function. */
			vector<bool> m_lineEndsOnCell;

			vector<DisconnectedRegion> m_disconnectedRegions;

			/** Location inside the grid */
			faceLocation_t m_faceLocation;
			dimensions_t m_faceDim;
			
			/** Vertices offset to facilitate searching inside vertices vector*/
			unsigned int m_edgeVerticesOffset;
			unsigned int m_insideVerticesOffset;
			unsigned int m_geometryEdgesOffset;
			/** Distance proximity constant */
			DoubleScalar m_distanceProximity;

			/** Vertex map */
			map<unsigned int, vector<edge_t *>> m_vertexMap;
			#pragma endregion

			#pragma region InitializationFunctions
			void initializeGridVertices();
			void initializeEdgeVertices();
			void initializeInsideVerticesAndEdges();
			#pragma endregion
			
			#pragma region PrivateFunctionalities
			//Connects edges disconnected regions with a simple edge connecting algorithm
			void connectDisconnectedRegions();
			void connectDisconnectedRegions2();

			void breadthFirstSearch(edge_t &edge, CutFace<Vector3D> *pCutFace, int regionTag, int prevEdgeIndex);

			//Finds a non-visited face from the faces vector
			int findNonVisitedEdge();

			int findVertexID(const Vector3D &point) {
				int vertexID = -1;
				for (int i = 0; i < m_vertices.size(); i++) {
					if (m_vertices[i] == point) {
						vertexID = i;
						return vertexID;
					}
				}
				return vertexID;
			}

			int signedDistanceTag(const Vector3D & point);

			/** Works for vectors (points) that lie in on of the faces of the Non-Manifold mesh */
			Vector2D convertToVec2(const Vector3D &vec);

			/** Chooses which edge to go when coming from an geometry edge, considering the direction and the two grid edges */
			edge_t * chooseEdge(const Vector2D &direction, int currEdgeID, int currVertexID, vector<edge_t *> vertexEdges);

			bool isInsideFace(const Vector3D & point, const CutFace<Vector3D> &face, faceLocation_t faceLocation);

			int crossesThroughGeometryFaces(const Vector3D &p1, const Vector3D &p2);

			/** Splits an edge that passes through the split point into two different edges. Returns the ID of the added
				vertex. Returns -1 if edge was found. */
			int splitEdge(const Vector3D &splitPoint);

			Vector3D findValidInteriorPoint(CutFace<Vector3D> *pCutFace);

			int getClosestOuterPointID(const Vector3D &point) const;

			template <class VectorType>
			DoubleScalar linearInterpolationWeight(const VectorType &position, const VectorType &p1, const VectorType &p2) {
				return (position - p1).length() / (p2 - p1).length();
			}
			#pragma endregion

			#pragma region AddingFunctions
			//Used to add vertices that are not on top of grid points. Returns the index of the newly added vertex
			unsigned int addVertex(const Vector3D &newVertex, unsigned int edgesOffset);

			//Searches for all vertices on the structure for the given vertex. Returns -1 if not found
			int findVertex(const Vector3D &newVertex);

			//Adding edges to the edge structure
			void addBorderEdge(unsigned int vertexID1, unsigned int vertexID2, edgeLocation_t edgeLocation);
			//Add edge on the back of the edges vector
			void addEdge(unsigned int vertexID1, unsigned int vertexID2, edgeLocation_t edgeLocation);
			//Inserts an edge in a specified position on the edges vector.
			void insertEdge(unsigned int vertexID1, unsigned int vertexID2, unsigned int edgePosition, edgeLocation_t edgeLocation);
			void removeEdge(unsigned int edgeID);
			//Goes through all subsequent edges and fix their IDs. Must be called after remove edge or insert edge
			void fixEdgesIDs();
			#pragma endregion

			#pragma region comparisonFunctions
			static bool compareEdges(pair<Scalar, unsigned int> a, pair<Scalar, unsigned int>b);
			static bool compareDisconnectedRegionsA(DisconnectedRegion a, DisconnectedRegion b);
			static bool compareDisconnectedRegionsB(DisconnectedRegion a, DisconnectedRegion b);
			#pragma endregion
		};
	}
}


#endif