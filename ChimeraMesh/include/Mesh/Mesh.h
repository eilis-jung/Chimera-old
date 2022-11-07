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

#ifndef __CHIMERA_MESH_H___
#define __CHIMERA_MESH_H___

#pragma once

#include "ChimeraCore.h"
#include "Mesh/Edge.h"
#include "Mesh/Face.h"
#include "Mesh/Volume.h"

namespace Chimera {
	using namespace Core;
	
	namespace Meshes {

		template <class VectorType, template <class> class ElementPrimitive>
		class Mesh {
		
		public:

			#pragma region Constructors
			/** Constructor with pre-built structures */
			Mesh(const vector<Vertex<VectorType> *> &vertices, const vector<ElementPrimitive<VectorType> *> &elementsPrimitive);

			/** Empty constructor */
			Mesh();
			#pragma endregion

			#pragma region AccessFunctions
			const vector<Vertex<VectorType> *> & getVertices() const {
				return m_vertices;
			}

			vector<Vertex<VectorType> *> & getVertices() {
				return m_vertices;
			}

			const VectorType & getCentroid() const {
				return m_centroid;
			}

			const vector<ElementPrimitive<VectorType> *> & getElements() const {
				return m_elements;
			}

			vector<ElementPrimitive<VectorType> *> & getElements() {
				return m_elements;
			}

			const ElementPrimitive<VectorType> * getElement(uint id) const {
				return m_elements[id];
			}

			ElementPrimitive<VectorType> * getElement(uint id) {
				return m_elements[id];
			}

			void setName(const string &gName) {
				m_name = gName;
			}

			bool drawMesh() const {
				return m_draw;
			}

			const vector<VectorType> & getPoints() const {
				return m_points;
			}

			vector<VectorType> & getPoints() {
				return m_points;
			}

			
			#pragma endregion

			#pragma region Functionalities
			virtual ElementPrimitive<VectorType> * getElement(const VectorType &position) { return m_elements[0]; }

			/** Replaces an element on the same */
			void replaceElement(ElementPrimitive<VectorType> *pReplacedElement, ElementPrimitive<VectorType> *pNewElement) {
				for (int i = 0; i < m_elements.size(); i++) {
					if (*m_elements[i] == *pReplacedElement) {
						m_elements[i] = pNewElement;
						delete pReplacedElement;
						m_elements.erase(m_elements.begin() + i);
						return;
					}
				}
			}

			#pragma endregion
		protected:

			#pragma region Class Members
			/** Mesh identification */
			uint m_ID;

			/** Unique mesh ID increment variable*/
			static uint m_currID;

			/** Mesh identification name*/
			string m_name;

			/** Vertices: vertices ID's are relative to this vector structure. */
			vector<Vertex<VectorType> *> m_vertices;
			
			/** Mesh centroid */
			VectorType m_centroid;

			/** Drawing attribute */
			bool m_draw;

			/** Mesh Elements:
				In a case of a planar mesh, elements are chosen to be faces;
				In a case of a volumetric mesh, elements are chosen to be volumes;
				In a case of a line mesh, elements are chosen to be edges. */
			vector<ElementPrimitive<VectorType> *> m_elements;


			/** Accelerator structure used for rendering purpose: stores all positions of the vertices inside a continuous vector */
			vector<VectorType> m_points;
			#pragma endregion

			#pragma region Functionalities
			void initializePoints() {
				m_points.resize(m_vertices.size());
				for (int i = 0; i < m_vertices.size(); i++) {
					m_points[i] = m_vertices[i]->getPosition();
				}
			}

			void flushInternalStructures() {
				m_vertices.clear();
				m_elements.clear();
				m_points.clear();
			}
			#pragma endregion

		};
	}

}
#endif