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

#ifndef __CHIMERA_LINE_MESH_H_
#define __CHIMERA_LINE_MESH_H_
#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "Mesh/Vertex.h"
#include "Mesh/Mesh.h"

namespace Chimera {

	namespace Meshes {

		/** Using Curiously Recurring Template Pattern https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern 
			for sharing explicit template specializations */
		template <class ChildT, class VectorT, bool isVector2>
		class LineMeshBase : public Mesh <VectorT, Edge> {
		public:

			#pragma region ExteriorClasses
			typedef struct params_t {
				/** Mesh points */
				vector<VectorT> initialPoints;
				/** Position relative to the centroid */
				VectorT position; // Relative to the mesh's centroid
				/** Central relative to the initial points */
				VectorT centroid;
				/** Scalar extrude along normals width */
				Scalar extrudeAlongNormalWidth;

				params_t() {
					extrudeAlongNormalWidth = 0.0f;
				}

				void updateCentroid() {
					centroid = VectorT();
					if (initialPoints.back() == initialPoints.front()) { //Closed mesh
						for (int i = 0; i < initialPoints.size() - 1; i++) {
							centroid += initialPoints[i];
						}
						centroid /= initialPoints.size() - 1;
					}
					else {
						for (int i = 0; i < initialPoints.size(); i++) {
							centroid += initialPoints[i];
						}
						centroid /= initialPoints.size();
					}
				}
			} params_t;
			#pragma endregion PublicClassDefitions

			#pragma region Constructors
			LineMeshBase(const params_t &lineMeshParams, dimensions_t gridDimensions = dimensions_t(0, 0, 0), Scalar gridDx = 0.0f) :
				m_regularGridPatches(gridDimensions) {
				m_gridDx = gridDx;
				m_hasUpdated = false;
				m_gridDimensions = gridDimensions;
				m_params = lineMeshParams;
				if(lineMeshParams.initialPoints.size())
					m_isClosedMesh = lineMeshParams.initialPoints.back() == lineMeshParams.initialPoints.front();
			}
			#pragma endregion

			#pragma region AccessFunctions
			params_t * getParams() {
				return &m_params;
			}

			bool isClosedMesh() {
				return m_isClosedMesh;
			}
			
			void updateCentroid() {
				m_centroid = VectorT();
				for (int i = 0; i < m_vertices.size(); i++) {
					m_centroid += m_vertices[i]->getPosition();
				} 
				if (m_vertices.size() > 0) {
					m_centroid /= m_vertices.size();
				}
			}

			const vector<uint> & getPatchesIndices(int i, int j) const {
				return m_regularGridPatches(i, j);
			}

			bool hasUpdated() const {
				return m_hasUpdated;
			}

			void setHasUpdated(bool hasUpdatedVar) {
				m_hasUpdated = hasUpdatedVar;
			}

			#pragma endregion

			#pragma region Functionalities
			virtual Edge<VectorT> * getElement(const VectorT &position) {
				return m_elements.front();
			}

			/** Mesh update */
			void update(Scalar dt) {

			}

			/** Checks if the linemesh intersects with given segment */
			bool segmentIntersection(const VectorT &v1, const VectorT &v2);

			/** Checks if the linemesh intersects with given segment */
			bool segmentIntersection(const VectorT &v1, const VectorT &v2, VectorT &intersectionPoint);
			
			void removeSmallEdges() {
				for (int j = 0; j < m_vertices.size() - 1;) {
					int nextJ = roundClamp<int>(j + 1, 0, m_vertices.size());
					DoubleScalar edgeDistance = (m_vertices[j]->getPosition() - m_vertices[nextJ]->getPosition()).length();
					if (edgeDistance == 0.0f) {
						m_vertices.erase(m_vertices.begin() + j);
					} else {
						j++;
					}
				}
			}
			#pragma endregion

		protected:

			#pragma region ClassMembers
			bool m_isClosedMesh;
			params_t m_params;

			/** Auxiliary structure used for organizing line-patches inside a regular grid. It stores local elements that are
				inside a regular grid that this line mesh is embedded. This. if it stores a greater than zero vector inside the 
				array, it means that regular-grid location contains a part of the line mesh (the specific part is indicated by
				the indices) */
			Array2D<vector<uint>> m_regularGridPatches;

			/** Grid Variables */
			dimensions_t m_gridDimensions;
			Scalar m_gridDx;

			bool m_hasUpdated; 
			#pragma endregion

			#pragma region InitializationFunctions
			void initializeEdges();
			virtual void initializeRegularGridPatches();
			void initializeVertexNormals();
			#pragma endregion

			#pragma region PrivateFunctionalities
			/** Compute crossings with the grid */
			/** This is not used for 3D-vectors, so these declarations are defined again on the template specialization below. */
			virtual void computeGridCrossings(Scalar dx, bool perturbPoints = true) { };
			virtual void extrudeAlongNormals() { };
			#pragma endregion
		};

		template<class VectorT, bool isVector2>
		class LineMeshT : public LineMeshBase<LineMeshT<VectorT, isVector2>, VectorT, isVector2>
		{
			public:
				LineMeshT(const params_t &lineMeshParams, dimensions_t gridDimensions = dimensions_t(0, 0, 0), Scalar gridDx = 0.0f) :
					LineMeshBase(lineMeshParams, gridDimensions, gridDx) {
					
				}
		};

		/** Using CRTP to share members and functions with vector3 declaration */
		template<class VectorT>
		class LineMeshT<VectorT, true> : public LineMeshBase<LineMeshT<VectorT, true>, VectorT, true> {
		public:
			LineMeshT(const params_t &lineMeshParams, dimensions_t gridDimensions = dimensions_t(0, 0, 0), Scalar gridDx = 0.0f, bool perturbPoints = true) :
				LineMeshBase(lineMeshParams, gridDimensions, gridDx) {
				if (m_gridDx == 0.0f) {
					for (int i = 0; i < m_params.initialPoints.size(); i++) {
						m_vertices.push_back(new Vertex<VectorT>(m_params.initialPoints[i], geometryVertex));
					}
					if (m_params.initialPoints.front() != m_params.initialPoints.back()) {
						m_vertices.front()->setVertexType(borderGeometryVertex);
						m_vertices.back()->setVertexType(borderGeometryVertex);
						m_isClosedMesh = false;
					}
				}
				else {
					if (m_params.extrudeAlongNormalWidth > 0) {
						extrudeAlongNormals();
					}
					computeGridCrossings(gridDx);
					initializeEdges();
					initializeRegularGridPatches();
				}
				
				removeSmallEdges();
				updateCentroid();
				initializePoints();
				initializeVertexNormals();
			}

			void updatePoints(const params_t &lineMeshParams, bool perturbPoints = true) {
				//Cleaning up internal structures
				for (int i = 0; i < m_regularGridPatches.getDimensions().x; i++) {
					for (int j = 0; j < m_regularGridPatches.getDimensions().y; j++) {
						m_regularGridPatches(i, j).clear();
					}
				}
				flushInternalStructures();
				

				m_params = lineMeshParams;
				
				
				/*if (m_params.extrudeAlongNormalWidth > 0) {
					extrudeAlongNormals();
				}*/
				
				computeGridCrossings(m_gridDx);
				if (perturbPoints)
					perturbOnGridLineEdges(m_gridDx);
				initializeEdges();
				initializeRegularGridPatches();
				removeSmallEdges();
				updateCentroid();
				initializePoints();
				initializeVertexNormals();

				m_hasUpdated = true;
			}
 

			void updatePoints(bool perturbPoints = true) {
				//Cleaning up internal structures
				for (int i = 0; i < m_regularGridPatches.getDimensions().x; i++) {
					for (int j = 0; j < m_regularGridPatches.getDimensions().y; j++) {
						m_regularGridPatches(i, j).clear();
					}
				}
				flushInternalStructures();

				/*if (m_params.extrudeAlongNormalWidth > 0) {
					extrudeAlongNormals();
				}*/
				computeGridCrossings(m_gridDx);
				if(perturbPoints)
					perturbOnGridLineEdges(m_gridDx);
				initializeEdges();
				initializeRegularGridPatches();
				removeSmallEdges();
				updateCentroid();
				initializePoints();
				initializeVertexNormals();

				m_hasUpdated = true;
			}
			protected:

			#pragma region PrivateFunctionalities
			virtual void computeGridCrossings(Scalar dx, bool perturbPoints = true) override;
			virtual void extrudeAlongNormals() override;
			virtual void perturbOnGridLineEdges(Scalar dx);
			#pragma endregion
		};

		/** Vector3 version of CRTP */
		template<class VectorT>
		class LineMeshT<VectorT, false> : public LineMeshBase<LineMeshT<VectorT, false>, VectorT, false> {
		public:
			LineMeshT(const params_t &lineMeshParams, faceLocation_t planeLocation, dimensions_t gridDimensions, Scalar gridDx) : LineMeshBase(lineMeshParams) {
				m_planeLocation = planeLocation;
				m_gridDimensions = gridDimensions;
				m_gridDx = gridDx;

				initializeVertices(m_params.initialPoints);
				/*if (planeLocation == XYFace) {
					computeGridCrossingsXY(m_gridDx);
				} else if(planeLocation == XZFace) {
					computeGridCrossingsXZ(m_gridDx);
				} else if (planeLocation == YZFace) {
					computeGridCrossingsYZ(m_gridDx);
				}*/
				initializeEdges();
				initializeRegularGridPatches();
				//Test if there's border edges
				if (m_params.initialPoints.front() != m_params.initialPoints.back()) {
					m_isClosedMesh = false;
				}
				updateCentroid();
				initializeVertexNormals();
			}


			LineMeshT(const vector<Vertex<VectorT> *> &vertices, const vector<Edge<VectorT> *> &edges, const faceLocation_t planeLocation, 
						dimensions_t gridDimensions, Scalar gridDx, bool closedMesh = true) : LineMeshBase(params_t()) {
				m_vertices = vertices;
				m_elements = edges;
				m_planeLocation = planeLocation;
				m_gridDimensions = gridDimensions;
				m_gridDx = gridDx;
				m_isClosedMesh = closedMesh;

				for (int i = 0; i < m_vertices.size(); i++) {
					m_params.initialPoints.push_back(m_vertices[i]->getPosition());
				}
				if (m_isClosedMesh) {
					m_params.initialPoints.push_back(m_params.initialPoints.front());
				}

				//initializeEdges();
				initializeRegularGridPatches();
				//Test if there's border edges
				
				updateCentroid();
				initializeVertexNormals();
			}

			LineMeshT(const params_t &lineMeshParams, faceLocation_t planeLocation) : LineMeshBase(lineMeshParams) {
				m_planeLocation = planeLocation;
				//m_gridDimensions = ;
				//m_gridDx = gridDx;

				initializeVertices(m_params.initialPoints);
				initializeEdges();
				initializeRegularGridPatches();
				//Test if there's border edges
				if (m_params.initialPoints.front() != m_params.initialPoints.back()) {
					m_isClosedMesh = false;
				}
				updateCentroid();
				initializeVertexNormals();
			}
			protected:

			#pragma region PrivateFunctionalities
				void initializeVertices(const vector<VectorT> &points);
				virtual void initializeRegularGridPatches();
				void computeGridCrossingsXZ(Scalar dx, bool perturbPoints = false);
				void computeGridCrossingsXY(Scalar dx, bool perturbPoints = false);
				void computeGridCrossingsYZ(Scalar dx, bool perturbPoints = false);

			#pragma endregion

			faceLocation_t m_planeLocation;
		};



		template <typename VectorT>
		using LineMesh = LineMeshT<VectorT, isVector2<VectorT>::value>;
	}
}

#endif
