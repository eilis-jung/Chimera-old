// Copyright (c) 2013 GeometryFactory (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
// You can redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// Licensees holding a valid commercial license may use this file in
// accordance with the commercial license agreement provided with the software.
//
// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
//
// $URL$
// $Id$
//
//
// Author(s)     : Ilker O. Yaz and Sebastien Loriot

#ifndef CGAL_POLYGON_MESH_SLICER_3_H
#define CGAL_POLYGON_MESH_SLICER_3_H

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_halfedge_graph_segment_primitive.h>
#include <CGAL/tuple.h>

#include <vector>
#include <set>

#include <boost/foreach.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <CGAL/internal/Polygon_mesh_slicer_3/Traversal_traits.h>
#include <CGAL/internal/Polygon_mesh_slicer_3/Axis_parallel_plane_traits.h>

#include <boost/variant.hpp>

#include <CGAL/boost/graph/split_graph_into_polylines.h>

namespace CGAL {

	/// \ingroup PkgPolygonMeshProcessing
	/// Function object that can compute the intersection of planes with
	/// a triangulated surface mesh.
	///
	/// \tparam TriangleMesh must be a model of `FaceGraph` and `HalfedgeListGraph`
	/// \tparam Traits must be a model of `AABBGeomTraits`
	/// \tparam VertexPointPmap is a model of `ReadablePropertyMap` with
	///         `boost::graph_traits<TriangleMesh>::%vertex_descriptor` as key and
	///         `Traits::Point_3` as value type
	/// \tparam AABBTree must be an instanciation of `CGAL::AABB_tree` able to handle
	///         the edges of TriangleMesh, having its `edge_descriptor` as primitive id.
	/// \tparam UseParallelPlaneOptimization if `true`, the code will use specific
	///         predicates and constructions in case the functor is called with a plane
	///         orthogonal to a frame axis, the non-null coefficient being 1 or -1.
	/// Depends on \ref PkgAABB_treeSummary
	/// \todo Shall we document more in details what is required?
	///       `Traits` must provide:
	///        - `Plane_3`
	///        - `Point_3`
	///        - `Segment_3`
	///        - `Oriented_side_3` with `Oriented_side operator()(Plane_3, Point_3)`
	///        - `Do_intersect_3` with` boost::optional<variant<Point_3,Segment_3> operator()(Plane_3,Segment_3)`
	///        - `Do_intersect_3` with bool operator()(Plane_3, Bbox_3)`
	///
	/// \todo If we keep the traits for plane orthogonal to a frame axis, `Traits` must also provide:
	///       - `FT`
	///       - `Construct_cartesian_const_iterator_3` with `Iterator operator()(Point_3)` `Iterator` being a random access iterator with `FT` as value type
	///       - `Construct_point_3` with `Point_3 operator()(FT,FT,FT)`; `Construct_source_3` with  `const Point_3& operator()(Segment_3)`
	///       - `Construct_target_3` with `const Point_3& operator()(Segment_3)`
	///
	/// \todo `_object()` functions must also be provided
	template<class TriangleMesh,
	class Traits,
	class VertexPointPmap = typename boost::property_map< TriangleMesh, vertex_point_t>::type,
	class AABBTree = AABB_tree<
		AABB_traits<Traits,
		AABB_halfedge_graph_segment_primitive<TriangleMesh> > >,
		bool UseParallelPlaneOptimization = true>
	class Polygon_mesh_slicer_3
	{
		/// Polygon_mesh typedefs
		typedef typename boost::graph_traits<TriangleMesh>               graph_traits;
		typedef typename graph_traits::vertex_descriptor            vertex_descriptor;
		typedef typename graph_traits::edge_descriptor                edge_descriptor;
		typedef typename graph_traits::halfedge_descriptor        halfedge_descriptor;
		typedef typename graph_traits::face_descriptor                face_descriptor;

		/// Geometric typedefs
		typedef typename Traits::Plane_3                                      Plane_3;
		typedef typename Traits::Segment_3                                  Segment_3;
		typedef typename Traits::Point_3                                      Point_3;
		typedef typename Traits::FT                                                FT;

		/// typedefs for internal graph to get connectivity of the polylines
		typedef boost::variant<vertex_descriptor, edge_descriptor>     AL_vertex_info;
		typedef boost::adjacency_list <
			boost::vecS,
			boost::vecS,
			boost::undirectedS,
			AL_vertex_info >                         AL_graph;
		typedef typename AL_graph::vertex_descriptor             AL_vertex_descriptor;
		typedef std::pair<AL_vertex_descriptor, AL_vertex_descriptor>  AL_vertex_pair;
		typedef std::map<vertex_descriptor, AL_vertex_descriptor>        Vertices_map;
		typedef std::pair<const vertex_descriptor, AL_vertex_descriptor>   Vertex_pair;
		/// Traversal traits
		typedef Slicer_helper::Traversal_traits<
			AL_graph,
			TriangleMesh,
			VertexPointPmap,
			typename AABBTree::AABB_traits,
			Traits >                                           General_traversal_traits;

		typedef Slicer_helper::Traversal_traits<
			AL_graph,
			TriangleMesh,
			VertexPointPmap,
			typename AABBTree::AABB_traits,
			Slicer_helper::Axis_parallel_plane_traits<Traits>
		>                                              Axis_parallel_traversal_traits;
		/// Auxiliary classes
		// compare the faces using the halfedge descriptors
		struct Compare_face{
			TriangleMesh& m_tmesh;
			Compare_face(TriangleMesh& tmesh)
				:m_tmesh(tmesh)
			{}

			bool operator()(halfedge_descriptor hd1, halfedge_descriptor hd2) const
			{
				return face(hd1, m_tmesh) < face(hd2, m_tmesh);
			}
		};

		typedef std::map< halfedge_descriptor, AL_vertex_pair, Compare_face > AL_edge_map;

		template <class OutputIterator, class Traits_>
		struct Polyline_visitor{
			AL_graph& al_graph;
			TriangleMesh& m_tmesh;
			const Plane_3& m_plane;
			VertexPointPmap m_vpmap;
			typename Traits_::Intersect_3 intersect_3;
			std::vector<typename TriangleMesh::Vertex_handle> m_vertexRing;
			OutputIterator out;

			Polyline_visitor(TriangleMesh& tmesh,
				AL_graph& al_graph,
				const Plane_3& plane,
				VertexPointPmap vpmap,
				const Traits_& traits,
				OutputIterator out)
				: al_graph(al_graph)
				, m_tmesh(tmesh)
				, m_plane(plane)
				, m_vpmap(vpmap)
				, intersect_3(traits.intersect_3_object())
				, out(out)
			{}

			std::vector< Point_3 > current_poly;
			void start_new_polyline()
			{
				current_poly.clear();
			}
			void add_node(AL_vertex_descriptor node_id)
			{
				AL_vertex_info v = al_graph[node_id];
				if (const vertex_descriptor* vd_ptr = boost::get<vertex_descriptor>(&v))
				{
					current_poly.push_back(get(m_vpmap, *vd_ptr));
					vertex_descriptor vertexDescriptor = boost::get<vertex_descriptor>(v);
					m_vertexRing.push_back(vertexDescriptor);
				}
				else
				{
					edge_descriptor ed = boost::get<edge_descriptor>(v);
					halfedge_descriptor hd = halfedge(ed, m_tmesh);
					TriangleMesh::Halfedge_handle edhandle = hd;
					TriangleMesh::Halfedge_handle edhandleNext = edhandle->next();
					TriangleMesh::Halfedge_handle edhandlePrev = edhandle->prev();
					typedef typename TriangleMesh::HalfedgeDS HDS;
					CGAL::HalfedgeDS_decorator<HDS> D(m_tmesh.hds());
					Segment_3 s(
						get(m_vpmap, source(ed, m_tmesh)),
						get(m_vpmap, target(ed, m_tmesh))
						);
					typename cpp11::result_of<typename Traits_::Intersect_3(Plane_3, Segment_3)>::type
						inter = intersect_3(m_plane, s);
					// CGAL_assertion( inter );
					const Point_3* pt_ptr = boost::get<Point_3>(&(*inter));
					Point_3 pt(*pt_ptr);
					if (pt_ptr && pt != s.source() && pt != s.target()) {
						current_poly.push_back(*pt_ptr);

						int currSplitVertex = m_tmesh.size_of_vertices();
						TriangleMesh::Halfedge_handle newEdge = D.split_vertex(edhandle->prev(), edhandle->opposite());
						newEdge->opposite()->vertex()->point() = Point_3(*pt_ptr);
						newEdge->opposite()->vertex()->id = currSplitVertex;
						m_vertexRing.push_back(newEdge->opposite()->vertex());
					} else {
						if (pt == s.source()) {
							current_poly.push_back(*pt_ptr);
							m_vertexRing.push_back(edhandle->prev()->vertex());
						}
						else if (pt == s.target()) {
							current_poly.push_back(*pt_ptr);
							m_vertexRing.push_back(edhandle->opposite()->vertex());
						}
						Segment_3 *pSeg = boost::get<Segment_3>(&(*inter));
						if (pSeg) {

						}
						else {
							cout << "Something else detected??" << endl;
						}
					}
				}
			}

			void add_edge(AL_vertex_descriptor source_id, AL_vertex_descriptor target_id) {
				AL_vertex_info vsource = al_graph[source_id];
				AL_vertex_info vtarget = al_graph[target_id];
				halfedge_descriptor hd = halfedge(boost::get<edge_descriptor>(vsource), m_tmesh);
				halfedge_descriptor hd2 = halfedge(boost::get<edge_descriptor>(vtarget), m_tmesh);

				typedef typename TriangleMesh::HalfedgeDS HDS;
				CGAL::HalfedgeDS_decorator<HDS> D(m_tmesh.hds());
				D.split_face(hd, hd2);
			}

			pair<typename TriangleMesh::Halfedge_handle, typename TriangleMesh::Halfedge_handle> findHalfedgePair(typename TriangleMesh::Vertex_handle &v1, typename TriangleMesh::Vertex_handle &v2) {
				pair<TriangleMesh::Halfedge_handle, TriangleMesh::Halfedge_handle> halfedgePair;
				TriangleMesh::Halfedge_around_vertex_circulator he1 = v1->vertex_begin();
				TriangleMesh::Halfedge_around_vertex_circulator he2 = v2->vertex_begin();
				do {
					do {
						if (he1->facet() == he2->facet()) {
							halfedgePair.first = he1;
							halfedgePair.second = he2;
							return halfedgePair;
						}
						++he2;
					} while (he2 != v2->vertex_begin());
					++he1;
				} while (he1 != v1->vertex_begin());

				return halfedgePair;
			}
			void add_halfedges_ring() {
				if (m_vertexRing.size()) {
					typedef typename TriangleMesh::HalfedgeDS HDS;
					CGAL::HalfedgeDS_decorator<HDS> D(m_tmesh.hds());
					double sqrDistance = 0;
					for (int i = 0; i < m_vertexRing.size() - 1;) {
						sqrDistance = squared_distance(m_vertexRing[i]->point(), m_vertexRing[i + 1]->point());
						if (sqrDistance < 1e-15) {
							m_vertexRing.erase(m_vertexRing.begin() + i);
						}
						else {
							i++;
						}
					}
					sqrDistance = squared_distance(m_vertexRing.back()->point(), m_vertexRing.front()->point());
					if (sqrDistance < 1e-15) {
						m_vertexRing.pop_back();
					}
					for (int i = 0; i < m_vertexRing.size() - 1; i++) {
						pair<TriangleMesh::Halfedge_handle, TriangleMesh::Halfedge_handle> halfedgePair = findHalfedgePair(m_vertexRing[i], m_vertexRing[i + 1]);
						if (halfedgePair.first != NULL && halfedgePair.second != NULL && halfedgePair.first != halfedgePair.second) {
							if (!halfedgePair.first->is_border() || !halfedgePair.second->is_border())
								D.split_face(halfedgePair.first, halfedgePair.second);
						}

					}
					pair<TriangleMesh::Halfedge_handle, TriangleMesh::Halfedge_handle> halfedgePair = findHalfedgePair(m_vertexRing.back(), m_vertexRing.front());
					if (halfedgePair.first != NULL && halfedgePair.second != NULL && halfedgePair.first != halfedgePair.second) {
						if (!halfedgePair.first->is_border() || !halfedgePair.second->is_border())
							D.split_face(halfedgePair.first, halfedgePair.second);
					}
				}
			}

			void end_polyline()
			{
				CGAL_assertion(!current_poly.empty());
				*out++ = current_poly;
			}
		};

		/// member variables
		const AABBTree* m_tree_ptr;
		TriangleMesh& m_tmesh;
		VertexPointPmap m_vpmap;
		Traits m_traits;
		bool m_own_tree;

		/// Convenience graph functions
		edge_descriptor next_edge(edge_descriptor ed) const
		{
			return edge(next(halfedge(ed, m_tmesh), m_tmesh), m_tmesh);
		}

		edge_descriptor next_of_opposite_edge(edge_descriptor ed) const
		{
			return edge(next(opposite(halfedge(ed, m_tmesh), m_tmesh), m_tmesh), m_tmesh);
		}

		face_descriptor opposite_face(edge_descriptor ed) const
		{
			return face(opposite(halfedge(ed, m_tmesh), m_tmesh), m_tmesh);
		}
		/// Other private functions
		/// handle edge insertion in the adjacency_list graph
		/// we add an edge betweem two edge_descriptor if they
		/// share a common facet
		void update_al_graph_connectivity(
			edge_descriptor ed,
			AL_vertex_descriptor vd,
			AL_edge_map& al_edge_map,
			AL_graph& al_graph) const
		{
			typename AL_edge_map::iterator itm;
			bool new_insertion;

			halfedge_descriptor hd = halfedge(ed, m_tmesh);

			if (face(hd, m_tmesh) != graph_traits::null_face())
			{
				cpp11::tie(itm, new_insertion) =
					al_edge_map.insert(std::pair< halfedge_descriptor, AL_vertex_pair >
					(hd, AL_vertex_pair(vd, AL_graph::null_vertex())));
				if (!new_insertion)
				{
					CGAL_assertion(itm->second.second == AL_graph::null_vertex());
					itm->second.second = vd;
					add_edge(itm->second.first,
						itm->second.second,
						al_graph);
				}
			}

			hd = opposite(hd, m_tmesh);
			if (face(hd, m_tmesh) != graph_traits::null_face())
			{
				cpp11::tie(itm, new_insertion) =
					al_edge_map.insert(std::pair< halfedge_descriptor, AL_vertex_pair >
					(hd, AL_vertex_pair(vd, AL_graph::null_vertex())));
				if (!new_insertion)
				{
					CGAL_assertion(itm->second.second == AL_graph::null_vertex());
					itm->second.second = vd;
					add_edge(itm->second.first,
						itm->second.second,
						al_graph);
				}
			}
		}

		std::pair<int, FT>
			axis_parallel_plane_info(const Plane_3& plane) const
		{
				FT a = m_traits.compute_a_3_object()(plane);
				FT b = m_traits.compute_b_3_object()(plane);
				FT c = m_traits.compute_c_3_object()(plane);
				FT d = m_traits.compute_d_3_object()(plane);

				if (a == 0)
				{
					if (b == 0)
					{
						if (c == 1 || c == -1)
							return std::pair<int, FT>(2, -d*c); /// z=-d
					}
					else
					{
						if (c == 0 && (b == 1 || b == -1))
							return std::pair<int, FT>(1, -d*b); /// y=-d
					}
				}
				else
				if (b == 0 && c == 0 && (a == 1 || a == -1))
					return std::pair<int, FT>(0, -d*a); /// x=-d
				return std::pair<int, FT>(-1, 0);
			}

	public:

		/**
		* Constructor using all `edges(tmesh)` to initialize the
		* internal `AABB_tree`.
		* @param tmesh the triangulated surface mesh to be cut.
		*              it must be valid and non modified as long
		*              as the functor is used
		* @param vpmap an intance of the vertex point property map
		* @param traits a traits class instance
		*/
		Polygon_mesh_slicer_3(const TriangleMesh& tmesh,
			VertexPointPmap vpmap,
			const Traits& traits = Traits())
			: m_tmesh(const_cast<TriangleMesh&>(tmesh))
			, m_vpmap(vpmap)
			, m_traits(traits)
			, m_own_tree(true)
		{
			m_tree_ptr = new AABBTree(edges(m_tmesh).first,
				edges(m_tmesh).second,
				m_tmesh,
				m_vpmap);
		}

		/**
		* Constructor using an `AABB_tree` provided by the user.
		* @param tmesh the triangulated surface mesh to be cut.
		*              it must be valid and non modified as long
		*              as the functor is used
		* @param tree must be initialized with all the edge of `tmesh`
		* @param vpmap an intance of the vertex point property map
		* @param traits a traits class instance
		*/
		Polygon_mesh_slicer_3(const TriangleMesh& tmesh,
			const AABBTree& tree,
			VertexPointPmap vpmap,
			const Traits& traits = Traits())
			: m_tree_ptr(&tree)
			, m_tmesh(const_cast<TriangleMesh&>(tmesh))
			, m_vpmap(vpmap)
			, m_traits(traits)
			, m_own_tree(false)
		{ }

		/**
		* Constructor using all `edges(tmesh)` to initialize the
		* internal `AABB_tree`. The vertex point property map used
		* is `get(boost::vertex_point, tmesh)`
		* @param tmesh the triangulated surface mesh to be cut.
		*              it must be valid and non modified as long
		*              as the functor is used
		* @param traits a traits class instance
		*/
		Polygon_mesh_slicer_3(const TriangleMesh& tmesh,
			const Traits& traits = Traits())
			: m_tmesh(const_cast<TriangleMesh&>(tmesh))
			, m_vpmap(get(boost::vertex_point, m_tmesh))
			, m_traits(traits)
			, m_own_tree(true)
		{
			m_tree_ptr = new AABBTree(edges(m_tmesh).first,
				edges(m_tmesh).second,
				m_tmesh,
				m_vpmap);
		}

		/**
		* Constructor using an `AABB_tree` provided by the user.
		* The vertex point property map used is `get(boost::vertex_point, tmesh)`
		* @param tmesh the triangulated surface mesh to be cut.
		*              it must be valid and non modified as long
		*              as the functor is used
		* @param tree must be initialized with all the edge of `tmesh`
		* @param traits a traits class instance
		*/
		Polygon_mesh_slicer_3(const TriangleMesh& tmesh,
			const AABBTree& tree,
			const Traits& traits = Traits())
			: m_tree_ptr(&tree)
			, m_tmesh(const_cast<TriangleMesh&>(tmesh))
			, m_vpmap(get(boost::vertex_point, m_tmesh))
			, m_traits(traits)
			, m_own_tree(false)
		{ }

		/**
		* Construct the intersecting polylines of `plane` with the input triangulated surface mesh.
		* @tparam OutputIterator an output iterator accepting polylines. A polyline is provided as `std::vector<Kernel::Point_3>`.
		*                        A polyline is closed if its first and last points are identical.
		* @param plane the plane to intersect the triangulated surface mesh with
		* @param out output iterator of polylines
		*/
		template <class OutputIterator>
		OutputIterator operator() (const Plane_3& plane,
			OutputIterator out) const
		{
			CGAL_precondition(!plane.is_degenerate());

			/// containers for storing edges wrt their position with the plane
			std::set<edge_descriptor> all_coplanar_edges;
			std::vector<edge_descriptor> iedges;
			Vertices_map vertices;

			/// get all edges intersected by the plane and classify them

			std::pair<int, FT> app_info = axis_parallel_plane_info(plane);
			if (!UseParallelPlaneOptimization || app_info.first == -1)
			{
				General_traversal_traits ttraits(
					all_coplanar_edges,
					iedges,
					vertices,
					m_tmesh,
					m_vpmap,
					m_tree_ptr->traits(),
					m_traits);
				m_tree_ptr->traversal(plane, ttraits);
			}
			else
			{
				Slicer_helper::Axis_parallel_plane_traits<Traits>
					traits(app_info.first, app_info.second, m_traits);

				Axis_parallel_traversal_traits ttraits(
					all_coplanar_edges,
					iedges,
					vertices,
					m_tmesh,
					m_vpmap,
					m_tree_ptr->traits(),
					traits);
				m_tree_ptr->traversal(plane, ttraits);
			}

			/// init output graph
			AL_graph al_graph;

			// add nodes for each vertex in the plane
			BOOST_FOREACH(Vertex_pair& vdp, vertices)
			{
				vdp.second = add_vertex(al_graph);
				al_graph[vdp.second] = vdp.first;
			}

			Compare_face less_face(m_tmesh);
			AL_edge_map al_edge_map(less_face);

			/// Filter coplanar edges: we consider only coplanar edges incident to one non-coplanar facet
			///   for each such edge, add the corresponding nodes in the adjacency-list graph as well as
			///   the edge
			BOOST_FOREACH(const edge_descriptor ed, all_coplanar_edges)
			{
				if (face(halfedge(ed, m_tmesh), m_tmesh) == graph_traits::null_face() ||
					opposite_face(ed) == graph_traits::null_face() ||
					!all_coplanar_edges.count(next_edge(ed)) ||
					!all_coplanar_edges.count(next_of_opposite_edge(ed)))
				{
					typename Vertices_map::iterator it_insert1, it_insert2;
					bool is_new;

					/// Each coplanar edge is connecting two nodes
					//  handle source
					cpp11::tie(it_insert1, is_new) =
						vertices.insert(
						Vertex_pair(
						source(ed, m_tmesh), AL_graph::null_vertex()
						)
						);
					if (is_new)
					{
						it_insert1->second = add_vertex(al_graph);
						al_graph[it_insert1->second] = it_insert1->first;
					}
					//  handle target
					cpp11::tie(it_insert2, is_new) =
						vertices.insert(
						Vertex_pair(
						target(ed, m_tmesh), AL_graph::null_vertex()
						)
						);
					if (is_new)
					{
						it_insert2->second = add_vertex(al_graph);
						al_graph[it_insert2->second] = it_insert2->first;
					}
					/// add the edge into the adjacency-list graph
					CGAL_assertion(it_insert1->second != AL_graph::null_vertex());
					CGAL_assertion(it_insert2->second != AL_graph::null_vertex());
					add_edge(it_insert1->second, it_insert2->second, al_graph);
				}
			}

			/// for each edge intersected in its interior, creates a node in
			/// an adjacency-list graph and put an edge between two such nodes
			/// when the corresponding edges shares a common face
			BOOST_FOREACH(edge_descriptor ed, iedges)
			{
				AL_vertex_descriptor vd = add_vertex(al_graph);
				al_graph[vd] = ed;
				update_al_graph_connectivity(ed, vd, al_edge_map, al_graph);
			}

			/// If one of the node above is not connected in its two incident faces
			/// then it must be connected to a vertex (including those in the set
			/// of coplanar edges)
			typedef std::pair<halfedge_descriptor, AL_vertex_pair> Halfedge_and_vertices;
			BOOST_FOREACH(Halfedge_and_vertices hnv, al_edge_map)
			{
				if (hnv.second.second == AL_graph::null_vertex())
				{
					//get the edge and test opposite vertices (if the edge is not on the boundary)
					vertex_descriptor vd = target(next(hnv.first, m_tmesh), m_tmesh);
					BOOST_FOREACH(Vertex_pair& vdp, vertices) {
						TriangleMesh::Halfedge_around_vertex_circulator he1 = vdp.first->vertex_begin();
						bool addedEdge = false;
						do {
							if (he1->facet() == hnv.first->facet() || he1->facet() == hnv.first->opposite()->facet()) {
								add_edge(vdp.second, hnv.second.first, al_graph);
							}
							++he1;
						} while (he1 != vdp.first->vertex_begin());

						if (addedEdge)
							break;

						/*if (vdp.first->halfedge()->facet() == hnv.first->facet() || vdp.first->halfedge()->opposite()->facet() == hnv.first->facet() ||
						vdp.first->halfedge()->facet() == hnv.first->opposite()->facet() || vdp.first->halfedge()->opposite()->facet() == hnv.first->opposite()->facet()) {
						add_edge(vdp.second, hnv.second.first, al_graph);
						}*/
						/*if (shareCommonFace(vdp.first, hnv.first)) {
						add_edge(vdp.second, hnv.second.first, al_graph);
						}*/
					}
					/*typename Vertices_map::iterator itv=vertices.find(vd);
					CGAL_assertion( itv!=vertices.end() );
					add_edge(itv->second, hnv.second.first, al_graph);*/

				}
			}

			CGAL_assertion(num_vertices(al_graph) == iedges.size() + vertices.size());

			/// now assemble the edges of al_graph to define polylines,
			/// putting them in the output iterator
			if (!UseParallelPlaneOptimization || app_info.first == -1)
			{
				Polyline_visitor<OutputIterator, Traits> visitor(m_tmesh, al_graph, plane, m_vpmap, m_traits, out);
				split_graph_into_polylines(al_graph, visitor);
				return visitor.out;
			}
			else
			{
				typedef Slicer_helper::Axis_parallel_plane_traits<Traits> App_traits;
				App_traits app_traits(app_info.first, app_info.second, m_traits);

				Polyline_visitor<OutputIterator, App_traits> visitor
					(m_tmesh, al_graph, plane, m_vpmap, app_traits, out);
				split_graph_into_polylines(al_graph, visitor);
				visitor.add_halfedges_ring();
				return visitor.out;
			}


		}

		~Polygon_mesh_slicer_3()
		{
			if (m_own_tree) delete m_tree_ptr;
		}
	};

}// end of namespace CGAL
#endif //CGAL_POLYGON_MESH_SLICER_3_H
