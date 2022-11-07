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

#ifndef __CHIMERA_CGAL_WRAPPER_UTILS_H_
#define __CHIMERA_CGAL_WRAPPER_UTILS_H_
#pragma once

#include "ChimeraCore.h"
#include "CGALConfig.h"
#include "Mesh/Volume.h"

namespace Chimera {

	using namespace Meshes;
	namespace CGALWrapper {
		
		FORCE_INLINE int searchVertex(Kernel::Point_3 needle, const vector<Vector3> &vertices) {
			for (int i = 0; i < vertices.size(); ++i) {
				if (vertices[i].x == needle.x() && vertices[i].y == needle.y() && vertices[i].z == needle.z())
					return i;
			}
		}

		void triangulatePolyhedron(CgalPolyhedron *pCGALPoly);


		// The BuildCgalPolyhedronFromObj class builds a CGAL::Polyhedron_3 from a list of vertices and polygons.
		// Faces can be polygons and doesn't have to be triangles.
		template<class HDS, class VectorType>
		class BuildCgalPolyhedronFromVertexList : public CGAL::Modifier_base<HDS> {
		public:

			BuildCgalPolyhedronFromVertexList(const vector<VectorType> &vertices, const vector<vector<pair<unsigned int, unsigned int>>> &faces)
				: m_vertices(vertices), m_faces(faces) {
			}

			void operator() (HDS& hds)
			{
				typedef typename HDS::Vertex   Vertex;
				typedef typename Vertex::Point Point;

				// Count the number of vertices and facets.
				// This is used to reserve memory in HDS.
				std::string _line;
				int _numVertices = m_vertices.size();
				int _numFacets = m_faces.size();

				// Postcondition: hds is a valid polyhedral surface.
				CGAL::Polyhedron_incremental_builder_3<HDS> B(hds, true);

				// Load the data from OBJ file to HDS.
				B.begin_surface(_numVertices, _numFacets, int((_numVertices + _numFacets - 2)*2.1), CGAL::Polyhedron_incremental_builder_3<HDS>::ABSOLUTE_INDEXING);
				int vertexID = 0;
				for (int i = 0; i < m_vertices.size(); i++) {
					HDS::Vertex_handle vh = B.add_vertex(Conversion::vecToPoint3(m_vertices[i]));
					vh->id = vertexID++;
				}
				for (int i = 0; i < m_faces.size(); i++) {
					B.begin_facet();
					int lastDiscontinuosPoint = -1;
					for (int j = 0; j < m_faces[i].size(); j++) {
						int currFaceID = m_faces[i][j].first;
						//if (j == 0)
						//	B.add_vertex_to_facet(m_faces[i][j].first);
						//else if (m_faces[i][j].first == m_faces[i][j - 1].second)
						//	B.add_vertex_to_facet(m_faces[i][j].first);
						//else {
						//	lastDiscontinuosPoint = j - 1;
						//	//A facet inside a facet?
						//	B.end_facet();
						//	B.begin_facet();
						//	B.add_vertex_to_facet(m_faces[i][j].first);
						//}
						B.add_vertex_to_facet(m_faces[i][j].first);
					}
					/*if (lastDiscontinuosPoint != -1) {
						B.add_vertex_to_facet(m_faces[i].back().second);
					}*/
					B.end_facet();
					
				}
				B.end_surface();
			}

		private:

			vector<vector<pair<unsigned int, unsigned int>>> m_faces;
			vector<VectorType> m_vertices;
		};



		template<class HDS, class VectorType>
		class BuildCgalPolyhedronFromVertexMap : public CGAL::Modifier_base<HDS> {
		public:

			BuildCgalPolyhedronFromVertexMap(HalfVolume<VectorType> *pHalfVolume)  {
				m_pHalfVolume = pHalfVolume;
			}

			void operator() (HDS& hds)
			{
				//typedef typename HDS::Vertex;
				//typedef typename Vertex::Point Point;

				const map<uint, Vertex<VectorType> *> &verticesMap = m_pHalfVolume->getVerticesMap();
				int _numVertices = verticesMap.size();
				int _numFacets = m_pHalfVolume->getHalfFaces().size();

				// Postcondition: hds is a valid polyhedral surface.
				CGAL::Polyhedron_incremental_builder_3<HDS> B(hds, true);
				B.begin_surface(_numVertices, _numFacets, int((_numVertices + _numFacets - 2)*2.1), CGAL::Polyhedron_incremental_builder_3<HDS>::ABSOLUTE_INDEXING);

				/** Add vertices first to the structure */
				map<uint, uint> verticesIDMap;
				uint vertexID = 0;
				for (auto iter = verticesMap.begin(); iter != verticesMap.end(); iter++) {
					HDS::Vertex_handle vh = B.add_vertex(Conversion::vecToPoint3(iter->second->getPosition()));
					verticesIDMap[iter->first] = vertexID;
					vh->id = iter->first;
				}
				
				for (uint i = 0; i < m_pHalfVolume->getHalfFaces().size(); i++) {
					const vector<HalfEdge<VectorType> *> &halfEdges = m_pHalfVolume->getHalfFaces()[i]->getHalfEdges();
					B.begin_facet();
					for (int j = 0; j < halfEdges.size(); j++) {
						uint vertexID2 = halfEdges[j]->getVertices().first->getID();
						uint vertexIDT = verticesIDMap[halfEdges[j]->getVertices().first->getID()];
						B.add_vertex_to_facet(verticesIDMap[halfEdges[j]->getVertices().first->getID()]);
					}
					B.end_facet();
				}
				B.end_surface();
			}

		private:
			HalfVolume<VectorType> *m_pHalfVolume;
		};



		// Converts vertices and faces representations to to polyhedron.
		// TPoly is a type of CGAL::Polyhdeon_3.
		template<class TPoly, class VectorT>
		void convertToPoly(const vector<VectorT> &vertices, const vector<vector<pair<unsigned int, unsigned int>>> &facesEdges, TPoly* polyhedron)
		{
			if (polyhedron)
			{
				try
				{
					// Build Polyhedron_3 from indices and vertices list
					BuildCgalPolyhedronFromVertexList<TPoly::HalfedgeDS, VectorT> _buildPolyhedron(vertices, facesEdges);

					// Calls is_valid at the end. Throws an exception in debug mode if polyhedron is not
					// manifold.
					polyhedron->delegate(_buildPolyhedron);

					// CGAL::Assert_exception is thrown in the debug mode when 
					// CGAL::Polyhedron_incremental_builder_3 is destroyed in BuildCgalPolyhedronFromObj.
					// However, in the release mode assertions is disabled and hence no exception is thrown.
					// Thus for uniform error reporting, if the polyhedron is not valid then throw a dummy 
					// exception in release mode.
					/*if (!polyhedron->is_valid())
					{
						throw CGAL::Assertion_exception("", "", "", 0, "");
					}*/
				}
				catch (const CGAL::Assertion_exception&)
				{
					std::string _msg = "Error converting vertices and indices to mesh";
					throw std::exception(_msg.c_str());
				}
			}
		}


		// Converts vertices and faces representations to to polyhedron.
		// TPoly is a type of CGAL::Polyhdeon_3.
		template<class TPoly, class VectorT>
		void convertToPoly(HalfVolume<VectorT> *pHalfVolume, TPoly* polyhedron)
		{
			if (polyhedron)
			{
				try
				{
					// Build Polyhedron_3 from indices and vertices list
					BuildCgalPolyhedronFromVertexMap<TPoly::HalfedgeDS, VectorT> _buildPolyhedron(pHalfVolume);

					// Calls is_valid at the end. Throws an exception in debug mode if polyhedron is not
					// manifold.
					polyhedron->delegate(_buildPolyhedron);

					// CGAL::Assert_exception is thrown in the debug mode when 
					// CGAL::Polyhedron_incremental_builder_3 is destroyed in BuildCgalPolyhedronFromObj.
					// However, in the release mode assertions is disabled and hence no exception is thrown.
					// Thus for uniform error reporting, if the polyhedron is not valid then throw a dummy 
					// exception in release mode.
					/*if (!polyhedron->is_valid())
					{
					throw CGAL::Assertion_exception("", "", "", 0, "");
					}*/
				}
				catch (const CGAL::Assertion_exception&)
				{
					std::string _msg = "Error converting vertices and indices to mesh";
					throw std::exception(_msg.c_str());
				}
			}
		}

		void translatePolygon(CgalPolyhedron *pPoly, const Vector3D &translationVec);
		void rotatePolygonZ(CgalPolyhedron *pPoly, DoubleScalar angle);
	}
}

#endif