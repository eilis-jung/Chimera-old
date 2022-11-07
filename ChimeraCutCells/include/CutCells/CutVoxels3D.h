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

#ifndef __CHIMERA_CUTVOXELS_3D_H__
#define __CHIMERA_CUTVOXELS_3D_H__

#pragma once

#include "ChimeraCore.h"
#include "CutCells/CutCells3D.h"
#include "ChimeraMesh.h"
#include "TriangleHalfVolume.h"

namespace Chimera {
	using namespace Meshes;

	namespace CutCells {
		
		template <class VectorType>
		class CutVoxels3D : public Mesh<VectorType, Volume> {
		
		public:
			#pragma region Constructors
			
			/** This constructor builds a planar mesh with multiple initialized line meshes. */
			CutVoxels3D(const vector<PolygonalMesh<VectorType> *> &polygonalMeshes, Scalar gridSpacing, const dimensions_t &gridDimensions);
			#pragma endregion

			#pragma region AccessFunctions
			/** Cut-Cell access functions */
			bool isCutVoxel(dimensions_t dimensions) {
				return m_volumesArray(dimensions) != nullptr;
			};
			bool isCutVoxel(int i, int j, int k) {
				return m_volumesArray(i, j, k) != nullptr;
			}
			/** Given a grid position, checks if it is a cut-cell. Position is in GRID SPACE */
			bool isCutVoxel(const VectorType &position) {
				return m_volumesArray(position.x, position.y, position.z) != nullptr;
			}

			uint getCutVoxelIndex(const VectorType &position);

			uint getNumberCutVoxels() const {
				return m_halfVolumes.size();
			}
			const HalfVolume<VectorType> & getCutVoxel(uint cellIndex) const {
				return *m_halfVolumes[cellIndex];
			}

			HalfVolume<VectorType> & getCutVoxel(uint cellIndex) {
				return *m_halfVolumes[cellIndex];
			}

			Volume<VectorType> * getVolume(dimensions_t dimensions) {
				return m_volumesArray(dimensions);
			}
			Volume<VectorType> * getVolume(uint i, uint j, uint k) {
				return m_volumesArray(i, j, k);
			}

			const map<uint, vector<LineMesh<VectorType> *>> & getXYLineMeshes() const {
				return m_XYLineMeshes;
			}

			const map<uint, vector<LineMesh<VectorType> *>> & getXZLineMeshes() const {
				return m_XZLineMeshes;
			}

			const map<uint, vector<LineMesh<VectorType> *>> & getYZLineMeshes() const {
				return m_YZLineMeshes;
			}


			const map<uint, CutCells3D<VectorType> *> & getXYCutCells() const {
				return m_XYCutCells;
			}

			const map<uint, CutCells3D<VectorType> *> & getXZCutCells() const {
				return m_XZCutCells;
			}

			const map<uint, CutCells3D<VectorType> *> & getYZCutCells() const {
				return m_YZCutCells;
			}

			Vertex<VectorType> * getNodalVertex(uint i, uint j, uint k) {
				return m_nodeVertices(i, j, k);
			}


			/** Faces access functions */
			const vector<Face<VectorType> *> & getFaceVector(dimensions_t edgeIndex, faceLocation_t faceType) const {
				if (faceType == XZFace) {
					return m_XZFaces(edgeIndex);
				}
				else if (faceType == YZFace) {
					return m_YZFaces(edgeIndex);
				} else if (faceType == XYFace) {
					return m_XYFaces(edgeIndex);
				}
				return vector<Face<VectorType> *>();
			}

			vector<Face<VectorType> *> & getFaceVector(dimensions_t edgeIndex, faceLocation_t faceType) {
				if (faceType == XZFace) {
					return m_XZFaces(edgeIndex);
				}
				else if (faceType == YZFace) {
					return m_YZFaces(edgeIndex);
				}
				else if (faceType == XYFace) {
					return m_XYFaces(edgeIndex);
				}
				return vector<Face<VectorType> *>();
			}

			/** Grid-spacing functions */
			DoubleScalar getGridSpacing() const {
				return m_gridSpacing;
			}

			const dimensions_t & getGridDimensions() const {
				return m_gridDimensions;
			}

			/** Line meshes */
			const vector<PolygonalMesh<VectorType> *> & getPolygonalMeshes() const {
				return m_polyMeshes;
			}
			#pragma endregion

			#pragma region Functionalities
			/** Reinitilize internal structures without deleting the pointer to this class*/
			void reinitialize(const vector<PolygonalMesh<VectorType> *> &polygonalMeshes);

			#pragma endregion
		protected:
			#pragma region ClassMembers
			/** Vector based structures facilitates global access.*/
			vector<HalfVolume<VectorType> *> m_halfVolumes;
			
			/** If initialized, triangle meshes */
			vector<TriangleHalfVolume<VectorType> *> m_triangleHalfVolumes;

			/** Polygonal meshes */
			vector<PolygonalMesh<VectorType> *> m_polyMeshes;

			/** Patches indices, accessible by the array structure. First index of the pair indicates the PolygonalMesh index,
				second is a vector containing all faces indices from a polygonal mesh that are contained inside the cell. */
			Array3D<vector<pair<uint, vector<uint>>>> m_polyPatches;

			/** Volumes are non-manifold non-oriented face structures. In order to initialize their half-volumes, that 
				correspond to cut-voxels on a grid, one has to individually split these half volumes 
				This is done locally by each volume. */
			Array3D<Volume<VectorType>*> m_volumesArray;

			/** Vertical and horizontal faces*/
			Array3D<vector<Face<VectorType> *>> m_XZFaces;
			Array3D<vector<Face<VectorType> *>> m_YZFaces;
			Array3D<vector<Face<VectorType> *>> m_XYFaces;

			/** Line meshes organized by slice index. Since multiple poly meshes are possible, we have to store the ones
			  * that are on the same slice on the same structure, so cut-cells can be properly initialized */
			map<uint, vector<LineMesh<VectorType> *>> m_XYLineMeshes;
			map<uint, vector<LineMesh<VectorType> *>> m_YZLineMeshes;
			map<uint, vector<LineMesh<VectorType> *>> m_XZLineMeshes;

			/** Cut-Slices: the first index indicates the index on the orthogonal dimension to the plane*/
			map<uint, CutCells3D<VectorType> *> m_XYCutCells;
			map<uint, CutCells3D<VectorType> *> m_YZCutCells;
			map<uint, CutCells3D<VectorType> *> m_XZCutCells;


			/** Edges: keeping horizontal and vertical names to match the 2-D version */
			/** X oriented edges */
			Array3D<vector<Edge<VectorType> *> *> m_horizontalEdges;
			/** Y oriented edges */
			Array3D<vector<Edge<VectorType> *> *> m_verticalEdges;
			/** Z oriented edges */
			Array3D<vector<Edge<VectorType> *> *> m_transversalEdges;
			/** Nodal vertices */
			Array3D<Vertex<VectorType> *> m_nodeVertices;

			/** Incorporated grid attributes */
			DoubleScalar m_gridSpacing;
			dimensions_t m_gridDimensions;
			#pragma endregion

			#pragma region PrivateFunctionalities
			void addFacesToVertices(HalfFace<VectorType> *pHalfFace);
			#pragma endregion
			#pragma region InitializationFunctions
			/** Initializes line patches inside each regular grid cell . Each pair added to polypatches stores the index
				of the current polymesh; the second vector stores the faces indices relative to that specific polymesh */
			void buildPolyPatches();


			/** Creates a single node vertex */
			FORCE_INLINE void createNodeVertex(const dimensions_t &cellIndex) {
				VectorType vertexPosition(cellIndex.x*m_gridSpacing, cellIndex.y*m_gridSpacing, cellIndex.z*m_gridSpacing);
				m_vertices.push_back(new Vertex<VectorType>(vertexPosition, gridVertex));
				m_nodeVertices(cellIndex) = m_vertices.back();
			}

			/** Creates vertices that are on top of grid nodes */
			void buildNodeVertices();

			/** Verifies if faces on the face patch have vertices that are on top of grid nodes */
			void buildOnNodeVerticesFacesPatch(const dimensions_t &cellIndex);

			/** Organize all cut-cells lines into a single vector ordered by the slice index*/
			void buildLineMeshes();
			
			/** Creates cut-slices from line meshes computed from intersections of polyMeshes with the grid*/
			void buildCutSlices();

			/** Adds regular grid edges to cut-voxels locations. These edges where not initialized by Cut-Cells 3-D on 
				build cut slices phase.*/
			void buildRegularGridEdges();
			
			Face<VectorType> * createRegularGridFace(uint i, uint j, uint k, faceLocation_t faceLocation);

			/** Add faces structures from cut-cells to Array3D of faces */
			void buildFaces();

			/** Creates all grid faces, which will be subdivided in a later step. */
			void buildVolumes();

			/** Builds half-faces, the cut-cells, based on initialized grid edges. */
			void buildHalfVolumes();

			/** Build ghost-vertices, responsible for managing different velocity data from different sides of the mesh*/
			void buildVertexAdjacencies();

			/** Build triangle half-volumes */
			void buildTriangleHalfVolumes();
			#pragma endregion
		};
	}

}
#endif