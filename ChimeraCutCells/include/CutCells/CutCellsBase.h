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

#ifndef __CHIMERA_CUT_CELLS_DATA__
#define __CHIMERA_CUT_CELLS_DATA__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraMesh.h"

namespace Chimera {
	using namespace Core;

	namespace CutCells {
		
		template <class VectorType>
		class CutCellsBase : public Mesh<VectorType, Face> {
			public:

			#pragma region External Structures
			// Encodes all necessary pointers and structures needed for class construction
			typedef struct extStructures_t {

				/** Edges arrays */
				Array2D<vector<Edge<VectorType> *> *> *pVerticalEdges;
				Array2D<vector<Edge<VectorType> *> *> *pHorizontalEdges;
				Array2D<Vertex<VectorType> *> *pNodeVertices;
				
				extStructures_t() {
					pVerticalEdges = nullptr;
					pHorizontalEdges = nullptr;
					pNodeVertices = nullptr;
				}
			};
			#pragma endregion

			#pragma region Constructors
				/** Default constructor: creates internal structures upon call */
				CutCellsBase(const vector<LineMesh<VectorType> *> &lineMeshes, Scalar gridSpacing, const dimensions_t &gridDimensions, faceLocation_t faceLocation = XYFace);
			
				/** Initialization constructor: initalizes structures upon call */
				CutCellsBase(const extStructures_t & extStructures, const vector<LineMesh<VectorType> *> &lineMeshes, Scalar gridSpacing, const dimensions_t &gridDimensions, faceLocation_t faceLocation = XYFace);

				virtual ~CutCellsBase() {

				}
			#pragma endregion



			#pragma region AccessFunctionss
			/** Cut-Cell access functions */
			virtual bool isCutCell(const dimensions_t &dimensions) {
				return (*m_pFacesArray)(dimensions).size() > 0;
			};

			bool isCutCellAt(int i, int j) {
				return m_facesArray(i, j).size() > 0;
			}

			Face<VectorType> * getFace(uint i, uint j) {
				return (*m_pFacesArray)(i, j).front();
			}


			/** Returns the cut-cell (half-face) for a given 2-D or 3-D position. Implemented by the base class.*/
			virtual uint getCutCellIndex(const VectorType &position) = 0;

			/** Cut-cells getters */
			uint getNumberCutCells() const {
				return m_halfFaces.size();
			}

			const HalfFace<VectorType> & getCutCell(uint cellIndex) const {
				return *m_halfFaces[cellIndex];
			}

			HalfFace<VectorType> & getCutCell(uint cellIndex) {
				return *m_halfFaces[cellIndex];
			}

			const vector<Face<VectorType> *> & getFaces(dimensions_t dimensions) const {
				return (*m_pFacesArray)(dimensions);
			}

			const vector<Face<VectorType> *> & getFaces(uint i, uint j) const {
				return (*m_pFacesArray)(i, j);
			}

			Face<VectorType> * getFace(dimensions_t dimensions) {
				return (*m_pFacesArray)(dimensions).front();
			}

			/** Edges access functions */
			const vector<Edge<VectorType> *> & getEdgeVector(dimensions_t edgeIndex, edgeType_t edgeType) const {
				if (edgeType == xAlignedEdge) {
					return *(*m_pHorizontalEdges)(edgeIndex);
				}
				else if (edgeType == yAlignedEdge) {
					return *(*m_pVerticalEdges)(edgeIndex);
				}
				return vector<Edge<VectorType> *>();
			}

			vector<Edge<VectorType> *> & getEdgeVector(dimensions_t edgeIndex, edgeType_t edgeType) {
				if (edgeType == xAlignedEdge) {
					return *(*m_pHorizontalEdges)(edgeIndex);
				}
				else if (edgeType == yAlignedEdge) {
					return *(*m_pVerticalEdges)(edgeIndex);
				}
				return vector<Edge<VectorType> *>();
			}


			/** Grid-spacing functions */
			DoubleScalar getGridSpacing() const {
				return m_gridSpacing;
			}

			const dimensions_t & getGridDimensions() const {
				return m_gridDimensions;
			}

			/** Line meshes */
			const vector<LineMesh<VectorType> *> & getLineMeshes() const {
				return m_lineMeshes;
			}
			#pragma endregion

			#pragma region Functionalities
			/** Reinitilize internal structures without deleting the pointer to this class*/
			void reinitialize(const vector<LineMesh<VectorType> *> &lineMeshes);

			virtual void initialize() = 0;
			#pragma endregion
			
			protected:

			#pragma region ClassMembers
			
			/*  These structures are pointers, which facilitate sharing and communication in 3-D.*/	
			/** Vector based structures facilitates global access.*/
			vector<HalfFace<VectorType> *> m_halfFaces;

			/** Line-meshes are the 2-D objects or 3-D planar intersection of a 3-D object and the grid. */
			vector<LineMesh<VectorType> *> m_lineMeshes;

			/** Incorporated grid attributes */
			DoubleScalar m_gridSpacing;
			dimensions_t m_gridDimensions;

			/** Slices attributes */
			uint m_sliceIndex;
			faceLocation_t m_faceLocation;

			/** Faces are non-manifold non-oriented edge structures. In order to initialize their half-faces, that
			    correspond to cut-cells on a plane, one has to individually split these faces. This is done locally
				by each Face. In 2-D there's a single face per grid location; in 3-D multiple faces are added to same 
				location, similar to edges. Since faces array are always on a 2-D plane, they will be stored like that
				even for 3-D Cut-cells classes. */
			Array2D<vector<Face<VectorType> *>> *m_pFacesArray;

			/** Patches indices, accessible by the array structure. First index of the pair indicates the LineMesh index,
			second is a vector containing all indices from that LineMesh that are contained inside the cell. This is
			always a 2-D structure, since it can be local to each patch index*/
			Array2D<vector<pair<uint, vector<uint>>>> *m_pLinePatches;
			
			#pragma region SharedStructures
			/** Possibly shared structures in 3-D*/
			/** Vertical and horizontal undirected edges */
			Array2D<vector<Edge<VectorType> *> *> *m_pVerticalEdges;
			Array2D<vector<Edge<VectorType> *> *> *m_pHorizontalEdges;

			/** Nodal vertices */
			Array2D<Vertex<VectorType> *> *m_pNodeVertices;
			#pragma endregion


			#pragma region AccessFacilitators
			/** Facilitate access to pointer members of the parent class */
			Array2D<vector<Face<VectorType>*>> &m_facesArray;
			Array2D<vector<Edge<VectorType> *> *> &m_verticalEdges;
			Array2D<vector<Edge<VectorType> *> *> &m_horizontalEdges;
			Array2D<Vertex<VectorType> *> &m_nodeVertices;
			#pragma endregion

			#pragma endregion

			/** These initialization functions are dimension-free (2-D or 3-D) */
			#pragma region InitializationFunctions
			Array2D<vector<pair<uint, vector<uint>>>> * createLinePatches(faceLocation_t faceLocation);

			/** Initializes line patches inside each regular grid cell */
			void buildLinePatches();

			/** Creates vertices that are on top of grid nodes */
			void buildNodeVertices();

			/** Creates all necessary grid edges. */
			void buildGridEdges();

			/** Creates all grid faces, which will be subdivided in a later step. */
			void buildFaces();

			/** Builds half-faces, the cut-cells, based on initialized grid edges. */
			void buildHalfFaces();

			/** Build ghost vertices for each side of the geometry */
			virtual void buildGhostVertices();

			/** Build vertices edge adjacency list */
			void buildVertexEdgesAdjacencyMaps();
			#pragma endregion

			#pragma region AuxiliaryHelperFunctions
			/** Check if a potential edge is aligned with geometry edges inside a cell. If it is, the fluid edge will not be added
				to the list of edges of a certain face. */
			FORCE_INLINE bool hasAlignedEdges(uint vertex1, uint vertex2, dimensions_t linePatchesDim, halfEdgeLocation_t edgeType) {
				dimensions_t nextDim(linePatchesDim);
				if (hasAlignedEdges(vertex1, vertex2, linePatchesDim)) {
					return true;
				}
				if (edgeType == bottomHalfEdge) { 
					nextDim.y -= 1;
				} else if (edgeType == topHalfEdge) {
					nextDim.y += 1;
				} else if (edgeType == leftHalfEdge) {
					nextDim.x -= 1;
				} else if (edgeType == rightHalfEdge) {
					nextDim.x += 1;
				}
				return hasAlignedEdges(vertex1, vertex2, nextDim);
			}

			bool hasAlignedEdges(uint vertex1, uint vertex2, dimensions_t linePatchesDim);
			
			FORCE_INLINE bool validVertex(uint i, uint j) {
				if (m_nodeVertices(i, j) == nullptr || m_nodeVertices(i, j)->getID() == UINT_MAX) {
					return false;
				}
				return true;
			}
			#pragma endregion

			#pragma region PureVirtualFunctions
			/** Creates a 2-D or 3-D vertex */
			virtual Vertex<VectorType> *createVertex(uint i, uint j) = 0;

			/** Passes the half-edge location, which only makes sense in 2-D. In this way the create edge function must
				realize where the location is relative to*/
			virtual Edge<VectorType> * createGridEdge(Vertex<VectorType> *pV1, Vertex<VectorType> *pV2, halfEdgeLocation_t halfEdgeLocation) = 0;

			/** Sorts the vertices on edges accordingly with their positions on space */
			virtual void sortVertices(vector<Vertex<VectorType> *> &bottomVertices, vector<Vertex<VectorType> *> &leftVertices, vector<Vertex<VectorType> *> &topVertices, vector<Vertex<VectorType> *> &rightVertices) = 0;
			
			/** Classifies vertices on 2-D or 3-D */
			virtual void classifyVertex(const dimensions_t &gridDim, Vertex<VectorType> *pVertex, vector<Vertex<VectorType> *> &bottomVertices, vector<Vertex<VectorType> *> &leftVertices,
																								  vector<Vertex<VectorType> *> &topVertices, vector<Vertex<VectorType> *> &rightVertices) = 0;

			#pragma endregion
		};
	}
}


#endif
#pragma once
