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
#ifndef __CHIMERA_VOLUME_H_
#define __CHIMERA_VOLUME_H_

#include "Mesh/Face.h"

namespace Chimera {

	namespace Meshes {

		template <class VectorType>
		class Volume;

		template <class VectorType>
		class HalfVolume {
			public:

				#pragma region Constructors
				/**Effortless constructor: all half-faces are already precomputed */
				HalfVolume(const vector<HalfFace<VectorType> *> &halfFaces, Volume<VectorType> *pVolume) : m_halfFaces(halfFaces) {
					m_pParentVolume = pVolume;
					m_ID = m_currID++;
					classifyHalfFaces();
					m_centroid = computeCentroid();
					initializeVerticesMap();

				}
				
				static void resetIDs() {
					m_currID = UINT_MAX;
				}
				#pragma endregion


				#pragma region Access Functions
				uint getID() const {
					return m_ID;
				}
				
				const VectorType & getCentroid() const {
					return m_centroid;
				}

				const vector<HalfFace<VectorType> *> & getHalfFaces() const {
					return m_halfFaces;
				}

				Volume<VectorType> * getVolume() {
					return m_pParentVolume;
				}
				
				const map<uint, Vertex<VectorType> *> & getVerticesMap() const {
					return m_verticesMap;
				}
				
				const map<uint, Vertex<VectorType> *> & getOnEdgeVerticesMap() const {
					return m_onEdgeVertices;
				}
				#pragma endregion

				#pragma region Operators
				FORCE_INLINE bool operator==(const HalfVolume<VectorType> &rhs) const {
					return m_ID == rhs.getID();
				}
				#pragma endregion

				#pragma region Functionalities
				bool isInside(const VectorType &position, VectorType direction = VectorType(0, 0, 1)) const;
				
				bool crossedThroughGeometry(const VectorType &v1, const VectorType &v2, VectorType &crossingPoint);
				#pragma endregion


			protected:

			#pragma region ClassMembers
			/** Halffaces */
			vector<HalfFace<VectorType> *> m_halfFaces;

			/** Parent volume */
			Volume<VectorType> *m_pParentVolume;

			/** Internal vertices quick access structure */
			map<uint, Vertex<VectorType> *> m_verticesMap;

			/** Internal mixed node vertices quick access structure */
			map<uint, Vertex<VectorType> *> m_onEdgeVertices;

			/** Centroid */
			VectorType m_centroid;

			/** ID vars */
			uint m_ID;
			static uint m_currID;
			#pragma endregion  

			#pragma region PrivateFunctionalities
			VectorType computeCentroid() const {
				VectorType centroid;
				return centroid;
			}

			/** Initializes vertices map accelerator structure */
			void initializeVerticesMap();
			
			/** Classify half-faces to be located correctly. Backfaces may become frontfaces (depending on their location
				relative to the volume), as leftFaces may become rightFaces and bottomFaces may become topFaces */
			void classifyHalfFaces();
			#pragma endregion  
		
		};


		/**  */
		template <class VectorType>
		class Volume {
		public:

			#pragma region Constructors
			/** Just here for compilation compatibility issues: not actually used */
			Volume() {
				m_ID = -1;
			}

			Volume(const vector<Face<VectorType> *> &faces, dimensions_t cellLocation, DoubleScalar gridDx) : m_faces(faces), m_edgeToFaceMap(faces) {
				m_ID = m_currID++;
				m_gridCellLocation = cellLocation;
				m_gridDx = gridDx;
			}

			static void resetIDs() {
				m_currID = 0;
			}
			#pragma endregion 

			#pragma region Functionalities
			/** Splits this volume into a series of half-volumes (cut-voxels). It uses edges orientations to make correct 3-D turns. *
				Initializes the result inside m_halfVolumes vector structure. Returns m_halfVolumes structure */
			const vector<HalfVolume<VectorType> *> & split();
			
			/** Looks for the half-volume that this point is contained */
			HalfVolume<VectorType> * getHalfVolume(const VectorType &position);
			#pragma endregion


			#pragma region AccessFunctions
			uint getID() const {
				return m_ID;
			}

			const vector<Face<VectorType> *> & getFaces() const {
				return m_faces;
			}

			const vector<HalfVolume<VectorType> *> & getHalfVolumes() {
				return m_halfVolumes;
			}

			void setGridCellLocation(const dimensions_t &gridCellLocation) {
				m_gridCellLocation = gridCellLocation;
			}
			const dimensions_t & getGridCellLocation() const {
				return m_gridCellLocation;
			}

			Scalar getGridSpacing() const {
				return m_gridDx;
			}

			#pragma endregion

			#pragma region Operators
			FORCE_INLINE bool operator==(const Volume<VectorType> &rhs) const {
				return m_ID == rhs.getID();
			}
			#pragma endregion

			
		protected:
			#pragma region ClassMembers
			/**Undirected faces of this volume */
			vector<Face<VectorType> *> m_faces;

			/** Interior split half volumes */
			vector<HalfVolume<VectorType> *> m_halfVolumes;

			/** Regular grid location and scale, if applicable */
			dimensions_t m_gridCellLocation;
			DoubleScalar m_gridDx;

			/** Each volume has a edge to edge face map that maps the elements of that specific Volume. This is a helper structure
				that selects the faces connected to the edge of this specific volume. This is useful when splitting the 
				volume into halfVolumes. */
			EdgeToFaceMap<VectorType> m_edgeToFaceMap;

			/** ID vars */
			uint m_ID;
			static uint m_currID;
			#pragma endregion

			#pragma region PrivateFunctionalities
			/** Search for unvisited faces. FaceID will contain the first non-visited element found. */
			Face<VectorType> * hasUnvisitedFaces() {
				for (uint i = 0; i < m_faces.size(); i++) {
					if (m_faces[i]->getLocation() != geometricFace && !m_faces[i]->isVisited()) {
						return m_faces[i];
					}
				}
				return nullptr;
			}

			/** Searches for half-faces inside a face, and returns the right one considering the dimension of the cell being evaluated. 
				Reverse edge gets back-facing geometry faces if true, otherwise returns front-facing geometry faces. */
			HalfFace<VectorType> * getHalfFace(Face<VectorType> *pFace,  dimensions_t cellLocation);

			/** Depth first search: in 3-D the connectivity graph does not branches, but the depth-first principle is 
				applied. This recursive function will call itself until it has visited all half-faces on a cycle. The visited
				half-faces will be appended to halfFaces vector. The reverse edge parameter tracks if the mesh went from
				the grid faces to geometry ones in a reversed manner, i.e. the common half-edge shared was a reversed one.
				If it was, then on the way back to grid-faces, the geometry edge has to be accessed on a reversed manner. */
			void breadthFirstSearch(HalfFace<VectorType> *pHalfFace, vector<HalfFace<VectorType> *> &halfFaces, HalfEdge<VectorType> *pPrevHalfEdge);
			#pragma endregion
		};

		template <class VectorType>
		unsigned int HalfVolume<VectorType>::m_currID = 0;

		template <class VectorType>
		unsigned int Volume<VectorType>::m_currID = 0;

	}
}

#pragma once
#endif