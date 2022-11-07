////  Copyright (c) 2013, Vinicius Costa Azevedo
////	All rights reserved.
////
////	Redistribution and use in source and binary forms, with or without
////	modification, are permitted provided that the following conditions are met: 
////
////1. Redistributions of source code must retain the above copyright notice, this
////	list of conditions and the following disclaimer. 
////	2. Redistributions in binary form must reproduce the above copyright notice,
////	this list of conditions and the following disclaimer in the documentation
////	and/or other materials provided with the distribution. 
////
////	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
////	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
////	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
////	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
////	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
////	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
////LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
////	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
////	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
////	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////
////	The views and conclusions contained in the software and documentation are those
////	of the authors and should not be interpreted as representing official policies, 
////	either expressed or implied, of the FreeBSD Project.
//
//
//#ifndef __CHIMERA_CGAL_WRAPPER_MESH_CUTTER_H_
//#define __CHIMERA_CGAL_WRAPPER_MESH_CUTTER_H_
//
//#pragma once
//#include "ChimeraCore.h"
//#include "ChimeraMath.h"
//#include "Mesh/PlaneMesh3D.h"
//#include "Mesh/TriangleMesh3D.h"
//#include "CGALConfig.h"
//
//namespace Chimera {
//
//	namespace CGALWrapper {
//
//		class MeshCutter : public Core::Singleton<MeshCutter> {
//		
//
//		private:
//
//		#pragma region PrivateClasses
//
//		template < class HDS, class Predicate>
//		class Polyhedron_cut_component_3 : public CGAL::Modifier_base<HDS> {
//		public:
//			typedef typename HDS::Halfedge_handle Halfedge_handle;
//		private:
//			Halfedge_handle h;
//			Predicate       pred;
//		public:
//			Polyhedron_cut_component_3(Halfedge_handle hh, const Predicate& p)
//				: h(hh), pred(p) {}
//			void operator()(HDS& target) {
//				h = halfedgeDSCutComponent(target, h, pred);
//			}
//			Halfedge_handle halfedge() const { return h; }
//		};
//
//		template < class Poly, class Plane, class Traits>
//		class I_Polyhedron_cut_plane_predicate {
//			const Plane&  m_plane;
//			const Traits& traits;
//		public:
//
//			typedef typename Poly::Vertex_const_handle Vertex_const_handle;
//			I_Polyhedron_cut_plane_predicate(const Plane& pl, const Traits& tr)
//				: m_plane(pl), traits(tr) {}
//			
//			bool operator()(Vertex_const_handle v) const {
//				return traits.has_on_negative_side_3_object()(m_plane, v->point());
//			}
//
//			const Plane& plane() const {
//				return m_plane;
//			}
//		};
//
//		#pragma endregion
//
//		#pragma region PrivateFunctionalities
//		// Cuts out a piece of a halfedge data structure for which the predicate
//		// `pred' is true for the vertices.
//		// The edge-vertex graph of the piece has to be a connected component.
//		// The remaining piece gets a new boundary. Returns a border halfedge of 
//		// the new boundary on the remaining piece. Assigns a halfedge of the 
//		// cut outpiece to `cut_piece'.
//		// The geometry for the vertices
//		// on the boundary and the hole have to be taken care of after this
//		// function call. The cut-out piece can be deleted with the member
//		// function erase_connected_component of the decorator class. It can
//		// technically happen that only an isolated vertex would remain in the
//		// cut out piece, in which case a dummy halfedge pair and vertex will be
//		// created to keep this vertex representable in the halfedge data structure.
//		// Precondition: pred( h->vertex()) && ! pred( h->opposite()->vertex()).
//		template < class HDS, class Predicate >
//		static typename HDS::Halfedge_handle halfedgeDSCutComponent(HDS& hds, typename HDS::Halfedge_handle h, Predicate pred, typename HDS::Halfedge_handle& cut_piece);
//
//		// Same function as above, but deletes the cut out piece immediately.
//		template < class HDS, class Predicate > 
//		static typename HDS::Halfedge_handle halfedgeDSCutComponent(HDS &hds, typename HDS::Halfedge_handle h, Predicate pred);
//
//		template < class Poly, class Predicate >
//		typename Poly::Halfedge_handle polyhedronCutComponent(Poly& poly, typename Poly::Halfedge_handle h, Predicate pred) {
//			typedef typename Poly::HalfedgeDS HDS;
//			typedef Polyhedron_cut_component_3<HDS, Predicate> Modifier;
//			Modifier modifier(h, pred);
//			HDS::Halfedge_handle hds(h);
//			poly.delegate(modifier);
//			return modifier.halfedge();
//		}
//		
//		// Cuts the polyhedron `poly' at plane `plane' starting at halfedge `h'.
//		// Traces the intersection curve of `plane' with `poly' starting at `h',
//		// cuts `poly' along that intersection curve and deletes the (connected)
//		// component on the negative side of the plane. The hole created along
//		// the intersection curve is filled with a new face containing the plane
//		// and the points in the vertices are computed.
//		template < class Poly, class Plane, class Traits>
//		typename Poly::Halfedge_handle polyhedronCutPlane(Poly& poly, typename Poly::Halfedge_handle h, const Plane& plane, const Traits& traits) {
//			typedef typename Poly::Halfedge_handle  Halfedge_handle;
//			typedef I_Polyhedron_cut_plane_predicate<Poly, Plane, Traits> Predicate;
//			Predicate pred(plane, traits);
//			CGAL_precondition(poly.is_valid());
//			CGAL_precondition(pred(h->vertex()));
//			CGAL_precondition(!pred(h->opposite()->vertex()));
//			h = polyhedronCutComponent(poly, h, pred);
//			CGAL_postcondition(poly.is_valid());
//			return h;
//		}
//
//		// Same function as above using the kernel that comes with the plane.
//		template < class Poly, class Plane>
//		typename Poly::Halfedge_handle polyhedronCutPlane(Poly& poly, typename Poly::Halfedge_handle h, const Plane& plane) {
//			typedef CGAL::Kernel_traits<Plane>  KTraits;
//			typedef typename KTraits::Kernel    Kernel;
//			return polyhedronCutPlane(poly, h, plane, Kernel());
//		}
//		#pragma endregion
//		
//		#pragma region ComparisonFunctions
//		static bool comparePairs(pair<CgalPolyhedron::Point_3, unsigned int *> a, pair<CgalPolyhedron::Point_3, unsigned int *> b);
//		static bool comparePairs2(pair<CgalPolyhedron::Point_3, unsigned int *> a, pair<CgalPolyhedron::Point_3, unsigned int *> b);
//		static bool comparePairs3(pair<Vector3, Vector3> a, pair<Vector3, Vector3> b);
//		static bool comparePairsInts(unsigned int * a, unsigned int * b);
//		#pragma endregion
//		public:
//
//		#pragma region Constructors
//		MeshCutter();
//		#pragma endregion
//
//
//		#pragma region Functionalities
//		/** Tests the half-space intersection between the plane and mesh. Everything that it's below
//		/** the plane normal is going to be deleted. New points are created on the intersection between
//		/** the plane and the mesh..*/
//		void halfspaceIntersection(CgalPolyhedron *pCgalPoly, const Data::PlaneMesh &plane);
//
//		/** This function corrects the errors introduced (by CGAL) on the halfspace intersection function. Returns
//		/** a pair of vertices and indices that can be used to initialize a PolygonSurface */
//		pair<vector<CgalPolyhedron::Point_3>, vector<vector<unsigned int>>> fixPolygon(CgalPolyhedron *pCgalPoly);
//
//		/** Returns an ordered set of points from a plane-polygon intersection */
//		vector<Vector3D> planeIntersection(CgalPolyhedron *pCgalPoly, const Data::PlaneMesh &plane);
//
//		/** Returns several intersections slices around the polygon mesh. */
//		vector<vector<Vector3D>> polygonSlicer(CgalPolyhedron *pCgalPoly, const Vector3 &normal, const Vector3 &origin, const Vector3 &increment, int numSlices);
//
//		/** Returns several intersections slices around multiple polygon meshes. */
//		vector<vector<Vector3D>> polygonSlicer(const vector<CgalPolyhedron *> &pCgalPolys, const Vector3 &normal, const Vector3 &origin, const Vector3 &increment, int numSlices);
//		#pragma endregion
//
//		}; //End class
//
//
//		template < class HDS, class Predicate >
//		typename HDS::Halfedge_handle MeshCutter::halfedgeDSCutComponent(HDS& hds, typename HDS::Halfedge_handle h, Predicate pred, typename HDS::Halfedge_handle& cut_piece) {
//			typedef typename HDS::Vertex_handle    Vertex_handle;
//			typedef typename HDS::Halfedge_handle  Halfedge_handle;
//			typedef typename HDS::Face_handle      Face_handle;
//			typedef typename HDS::Vertex           Vertex;
//			typedef typename HDS::Halfedge         Halfedge;
//			typedef typename HDS::Face             Face;
//			typedef typename Kernel::Line_3           Line;
//			typedef typename Kernel::Point_3          Point;
//			CGAL::HalfedgeDS_decorator<HDS> D(hds);
//
//			CGAL_precondition(D.is_valid(false, 3));
//			CGAL_precondition(pred(h->vertex()));
//			CGAL_precondition(!pred(h->opposite()->vertex()));
//
//			Halfedge_handle start = h;
//			Halfedge_handle hnew;
//			Halfedge_handle hlast;
//			while (true) {
//				// search re-entry point
//				Halfedge_handle g = h;
//				while (pred(g->next()->vertex())) {
//					g = g->next();
//					// create border edges around cap
//					D.set_face(g, Face_handle());
//				}
//				if (hnew == Halfedge_handle()) {
//					// first edge, special case
//					CGAL_assertion(g->next() != h && g->next()->opposite() != h);
//					Halfedge_handle gnext = g->next()->opposite();
//					D.remove_tip(g);
//					
//					// Compute intersection
//					Line line(h->vertex()->point(), h->opposite()->vertex()->point());
//					CGAL::Object obj = intersection(line, pred.plane());
//					Point pt;
//					Line otherLine;
//					Vertex_handle v;
//					if (assign(pt, obj)) {
//						v = D.vertices_push_back(Vertex(pt));
//					}
//					else { // this results in a line
//						v = D.vertices_push_back(Vertex(pred.plane().projection(g->vertex()->point())));
//					}
//					
//					D.close_tip(gnext, v);
//					hnew = hds.edges_push_back(Halfedge(), Halfedge());
//					hlast = hnew->opposite();
//					D.insert_tip(hlast, gnext);
//					D.set_face(hnew, D.get_face(gnext));
//					D.set_face_halfedge(hnew);
//					h = g;
//					D.set_vertex_halfedge(h);
//				}
//				else { // general case and last case
//					Halfedge_handle gnext = g->next()->opposite();
//					if (gnext == start && gnext == g) {
//						// last edge, special case of isolated vertex.
//						// Create dummy edge and dummy vertex and attach it to g
//						g = hds.edges_push_back(Halfedge(), Halfedge());
//						D.insert_tip(g, gnext);
//
//						// Compute intersection
//						Line line(h->vertex()->point(), h->opposite()->vertex()->point());
//						CGAL::Object obj = intersection(line, pred.plane());
//						Point pt;
//						Line otherLine;
//						Vertex_handle v;
//						if (assign(pt, obj)) {
//							v = D.vertices_push_back(Vertex(pt));
//						}
//						else { // this results in a line
//							v = D.vertices_push_back(Vertex(pred.plane().projection(g->vertex()->point())));
//						}
//
//						D.close_tip(g->opposite(), v);
//						D.set_vertex_halfedge(g);
//						D.set_vertex_halfedge(g->opposite());
//					}
//					D.remove_tip(g);
//					
//					// Compute intersection
//					Line line(h->vertex()->point(), h->opposite()->vertex()->point());
//					CGAL::Object obj = intersection(line, pred.plane());
//					Point pt;
//					Line otherLine;
//					Vertex_handle v;
//					if (assign(pt, obj)) {
//						v = D.vertices_push_back(Vertex(pt));
//					}
//					else { // this results in a line
//						v = D.vertices_push_back(Vertex(pred.plane().projection(g->vertex()->point())));
//					}
//
//					D.close_tip(hnew, v);
//					D.insert_tip(gnext, hnew);
//					hnew = hds.edges_push_back(Halfedge(), Halfedge());
//					D.insert_tip(hnew->opposite(), gnext);
//					D.set_face(hnew, D.get_face(gnext));
//					D.set_face_halfedge(hnew);
//					h = g;
//					D.set_vertex_halfedge(h);
//					if (gnext == start) {
//						// last edge, special
//						D.insert_tip(hnew, hlast);
//						break;
//					}
//				}
//			} // while(true)
//			Face_handle fnew = D.faces_push_back(Face());
//			D.set_face_in_face_loop(hlast, fnew);
//			D.set_face_halfedge(hlast);
//			cut_piece = h;
//			CGAL_postcondition(D.is_valid(false, 3));
//			return hlast;
//		}
//
//		template < class HDS, class Predicate > typename HDS::Halfedge_handle MeshCutter::halfedgeDSCutComponent(HDS &hds, typename HDS::Halfedge_handle h, Predicate pred) {
//			typedef typename HDS::Halfedge_handle  Halfedge_handle;
//			CGAL::HalfedgeDS_decorator<HDS> D(hds);
//
//			CGAL_precondition(D.is_valid(false, 3));
//			CGAL_precondition(pred(h->vertex()));
//			CGAL_precondition(!pred(h->opposite()->vertex()));
//			Halfedge_handle cut_piece;
//			Halfedge_handle hnew = halfedgeDSCutComponent(hds, h, pred, cut_piece);
//			D.erase_connected_component(cut_piece);
//			CGAL_postcondition(D.is_valid(false, 3));
//			return hnew;
//		}
//	}
//}
//#endif