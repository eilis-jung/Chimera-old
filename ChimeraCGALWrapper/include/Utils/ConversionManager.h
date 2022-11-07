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

#ifndef __CHIMERA_CGAL_WRAPPER_CONVERSION_MANAGER_H_
#define __CHIMERA_CGAL_WRAPPER_CONVERSION_MANAGER_H_

#pragma once
#include "ChimeraCore.h"
//#include "Mesh/PlaneMesh3D.h"
//#include "Mesh/TriangleMesh3D.h"
#include "Mesh/Mesh.h"
#include "Mesh/Face.h"
#include "CGALConfig.h"


namespace Chimera {
	using namespace Core;
	using namespace Meshes;

	typedef struct simpleFace_t {
		vector<pair<unsigned int, unsigned int>> edges;
		bool borderFace;
		Vector3D centroid;
		Vector3D normal;
		simpleFace_t() {
			borderFace = false;
		}

		//Check if this face shares an edge with the other simple face
		//Also considers topology, the edge is only valid if one is the
		//inverse of the other
		bool isConnectedTo(const simpleFace_t &other);
	} simpleFace_t;

	namespace CGALWrapper {

		/** Converts between Chimera base classes and CGAL classes.
		/**	Name convention: 
			* Vec: Chimera vector class. Can be either Vector2 or Vector3
			* Vec3: CGAL Vector class 
			* Point3: CGAL Point class
			(Basically everything that has an integer number after the name,
			 its a CGAL class)
		*/

		namespace Conversion {

			

			#pragma region VectorConversions
			template <class VectorType>
			FORCE_INLINE Kernel::Point_3 vecToPoint3(const VectorType &vec) {
				Kernel::Point_3 point(vec.x, vec.y, vec.z);
				return point;
			}

			template <class VectorType>
			FORCE_INLINE VectorType pointToVec(const Kernel::Point_3 &point) {
				return VectorType(static_cast<DoubleScalar>(point.x()), static_cast<DoubleScalar>(point.y()), static_cast<DoubleScalar>(point.z()));
			}

			template <class VectorType>
			FORCE_INLINE Kernel::Vector_3 vecToVec3(const VectorType &vec) {
				Kernel::Vector_3 temp(vec.x, vec.y, vec.z);
				return temp;
			}

			template <class VectorType>
			FORCE_INLINE VectorType vec3ToVec(const Kernel::Vector_3 &vec) {
				return VectorType(static_cast<DoubleScalar>(vec.x()), static_cast<DoubleScalar>(vec.y()), static_cast<DoubleScalar>(vec.z()));
			}
			#pragma endregion
			FORCE_INLINE uint edgeHash(uint i, uint j, uint verticesSize) {
				if (i < j)
					return verticesSize*i + j;
				return verticesSize*j + i;
			}

			//
			//#pragma region GeneralConversions
			//FORCE_INLINE Kernel::Plane_3 planeMeshToPlane3(const Data::PlaneMesh &planeMesh) {
			//	Kernel::Plane_3 plane(vecToPoint3(convertToVector3D(planeMesh.getOrigin())), vecToVec3(convertToVector3D(planeMesh.getNormal())));
			//	return plane;
			//}

			/** Converts from CgalPolyhedron to a vertices and faces vectors
			/*  Returns the total amount of vertices (face sub elements) of the output mesh*/
			template<class VectorType>
			int polyhedron3ToHalfFacesAndVertices(CgalPolyhedron *pPoly, vector<Vertex<VectorType> *> &vertices, vector<HalfFace<VectorType> *> &faces);


			template<class VectorType>
			int polyhedron3ToFacesAndVertices(CgalPolyhedron *pPoly, vector<Vertex<VectorType> *> &vertices, vector<Face<VectorType> *> &faces);

			template<class VectorType>
			int polyhedron3ToHalfFaces(CgalPolyhedron *pPoly, const map<uint, Vertex<VectorType> *> &verticesMap, vector<HalfFace<VectorType> *> &halffaces);

			//#pragma endregion

			//template<class TPoly>
			//void polyToTriangleMesh(TPoly* polyhedron, vector<Vector3D> &vertices, vector<Data::TriangleMesh::triangle_t> &triangles) {
			//	//Fixing vertices ID again
			//	int tempIndex = 0;
			//	
			//	for (auto it = polyhedron->vertices_begin(); it != polyhedron->vertices_end(); it++) {
			//		TPoly::Vertex_handle vh(it);
			//		vh->id = tempIndex++;
			//		vertices.push_back(pointToVec<Vector3D>(it->point()));
			//	}

			//	for (TPoly::Facet_iterator it = polyhedron->facets_begin(); it != polyhedron->facets_end(); ++it) {
			//		TPoly::Facet_handle fit = it;
			//		auto hfc = it->facet_begin();

			//		Data::TriangleMesh::triangle_t triangle;
			//		triangle.normal = fit->normal;
			//		Vector3D centroid;
			//		for (int j = 0; j < it->size(); ++j, ++hfc) {
			//			CgalPolyhedron::Vertex_handle vh(hfc->vertex());
			//			triangle.pointsIndexes[j] = vh->id;
			//			centroid += vertices[vh->id];
			//		}
			//		centroid /= it->size(); //presumably it->size = 3
			//		triangle.centroid = centroid;
			//		triangles.push_back(triangle);
			//	}
			//}

			//template<class TPoly>
			//void polyToMesh3D(TPoly* polyhedron, const vector<Vector3> &vertices, vector<typename Data::Mesh3D<Vector3>::meshPolygon_t> &polygons) {
			//	for (TPoly::Facet_iterator it = polyhedron->facets_begin(); it != polyhedron->facets_end(); ++it) {
			//		TPoly::Facet_handle fit = it;
			//		auto hfc = it->facet_begin();

			//		typename Data::Mesh3D<Vector3>::meshPolygon_t currPolygon;
			//		currPolygon.normal = convertToVector3F(fit->normal);
			//		Vector3 centroid;
			//		for (int j = 0; j < it->size(); ++j, ++hfc) {
			//			CgalPolyhedron::Vertex_handle vh(hfc->vertex());
			//			currPolygon.edges.push_back(pair<unsigned int, unsigned int>(vh->id, vh->id));
			//			centroid += vertices[vh->id];
			//		}
			//		for (int j = 0; j < currPolygon.edges.size(); j++) {
			//			int nextJ = roundClamp<int>(j + 1, 0, currPolygon.edges.size());
			//			currPolygon.edges[j].second = currPolygon.edges[nextJ].first;
			//		}
			//		centroid /= it->size(); //presumably it->size = 3
			//		currPolygon.centroid = centroid;
			//		polygons.push_back(currPolygon);
			//	}
			//}
			
			//template<class TPoly, class VectorType>
			//void polyToMesh(TPoly* polyhedron, vector<VectorType> &vertices, vector<typename Meshes::Mesh<VectorType>::meshPolygon_t> &polygons) {
			//	int tempIndex = 0;
			//	for (auto it = polyhedron->vertices_begin(); it != polyhedron->vertices_end(); it++) {
			//		TPoly::Vertex_handle vh(it);
			//		vh->id = tempIndex++;
			//		vertices.push_back(pointToVec<VectorType>(it->point()));
			//	}

			//	for (TPoly::Facet_iterator it = polyhedron->facets_begin(); it != polyhedron->facets_end(); ++it) {
			//		TPoly::Facet_handle fit = it;
			//		auto hfc = it->facet_begin();

			//		typename Meshes::Mesh<VectorType>::meshPolygon_t currPolygon;
			//		currPolygon.normal.x = fit->normal.x;
			//		currPolygon.normal.y = fit->normal.y;
			//		currPolygon.normal.z = fit->normal.z;
			//		VectorType centroid;
			//		for (int j = 0; j < it->size(); ++j, ++hfc) {
			//			CgalPolyhedron::Vertex_handle vh(hfc->vertex());
			//			currPolygon.edges.push_back(pair<unsigned int, unsigned int>(vh->id, vh->id));
			//			centroid += vertices[vh->id];
			//		}
			//		for (int j = 0; j < currPolygon.edges.size(); j++) {
			//			int nextJ = roundClamp<int>(j + 1, 0, currPolygon.edges.size());
			//			currPolygon.edges[j].second = currPolygon.edges[nextJ].first;
			//		}
			//		centroid /= it->size(); //presumably it->size = 3
			//		currPolygon.centroid = centroid;
			//		polygons.push_back(currPolygon);
			//	}
			//}
		};
		  
	}
}
#endif