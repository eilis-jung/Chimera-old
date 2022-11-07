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

#ifndef __M_CHIMERA_CGAL_CFG_H_
#define __M_CHIMERA_CGAL_CFG_H_
#pragma once

#include "ChimeraCore.h"

/************************************************************************/
/* CGAL                                                                 */
/************************************************************************/
#pragma warning (disable:4503)
#pragma warning (push, 1)
#include "CGAL/Simple_cartesian.h"
#include "CGAL/Exact_predicates_exact_constructions_kernel_with_sqrt.h"
#include "CGAL/Exact_predicates_inexact_constructions_kernel.h"
#include "CGAL/Mesh_3/Robust_intersection_traits_3.h"
#include "CGAL/Object.h"
#include "CGAL/Polyhedron_items_3.h"
#include "CGAL/HalfedgeDS_list.h"
#include "CGAL/HalfedgeDS_vector.h""
#include "CGAL/Polyhedron_3.h"
#include "CGAL/IO/Polyhedron_iostream.h"
#include "CGAL/Polyhedron_incremental_builder_3.h"
#include "CGAL/Modifier_base.h"
#include "CGAL/exceptions.h"
#include "CGAL/Simple_cartesian.h"
#include "CGAL/AABB_tree.h"
#include "CGAL/AABB_traits.h"
#include "CGAL/Polyhedron_3.h"
#include "CGAL/boost/graph/graph_traits_Polyhedron_3.h"
#include "CGAL/AABB_face_graph_triangle_primitive.h"
#include "CGAL/AABB_halfedge_graph_segment_primitive.h"
#include "CGAL/Modifier_base.h"
#include "CGAL/triangulate_polyhedron.h"
#include "CGAL/Polygon_mesh_slicer.h"
#include "CGAL/Polygon_mesh_slicer_3.h"
#include "CGAL/property_map.h"
#include <CGAL/Unique_hash_map.h>
#include <CGAL/Mesh_3/dihedral_angle_3.h>


#include <CGAL/boost/graph/graph_traits_Polyhedron_3.h>
// Simplification function
#include "CGAL/Surface_mesh_simplification/edge_collapse.h"
// Midpoint placement policy
#include "CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Midpoint_placement.h"
//Placement wrapper
#include "CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Constrained_placement.h"
// Stop-condition policy
#include "CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_stop_predicate.h"
#include "CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h"

#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/graph_traits_Surface_mesh.h>

#pragma warning (pop)

/** Define suitable CGAL names that will be used through the implementation */

namespace Chimera {
	using namespace Core;
	namespace CGALWrapper {

		/*namespace kernel_type_h {
			typedef CGAL::Exact_predicates_inexact_constructions_kernel K1;
		}*/

		//typedef CGAL::Mesh_3::Robust_intersection_traits_3<kernel_type_h::K1> Kernel; 
		typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
		//typedef CGAL::Simple_cartesian<float> Kernel;


		// A face type with a color member variable.
		template <class Refs> 
		struct CustomFace : public CGAL::HalfedgeDS_face_base<Refs> {    
			CustomFace() {
			}
			Vector3D normal;
		};
		template <class Refs, class Point>
		struct CustomVertex : public CGAL::HalfedgeDS_vertex_base<Refs, CGAL::Tag_true, Point> {
			CustomVertex() {
				cutSlicePointCount = -1;
			}
			CustomVertex(const Point& pt) : CGAL::HalfedgeDS_vertex_base<Refs, CGAL::Tag_true, Point>(pt) {
				cutSlicePointCount = -1;
			}
			uint id;
			int cutSlicePointCount;
		};

		template < class Refs >
		struct CustomEdge : public CGAL::HalfedgeDS_halfedge_base < Refs > {
			typedef CGAL::HalfedgeDS_halfedge_base<Refs> Halfedge;
			bool isCrossingEdge;

			CustomEdge() {
				isCrossingEdge = false;
			}
		};

		// An items type using my face and vertex.
		struct CustomItems : public CGAL::Polyhedron_items_3 {
			template <class Refs, class Traits>
			struct Face_wrapper {
				typedef CustomFace<Refs> Face;
			};

			template <class Refs, class Traits>
			struct Vertex_wrapper {
				typedef typename Traits::Point_3 Point;
				typedef 
					CustomVertex<Refs, Point> Vertex;
			};
			template <class Refs, class Traits>
			struct Halfedge_wrapper {
				typedef CustomEdge<Refs> Halfedge;
			};
		};

		typedef CGAL::Polyhedron_3<Kernel, CustomItems, CGAL::HalfedgeDS_list> CgalPolyhedron;

		typedef CGAL::AABB_halfedge_graph_segment_primitive<CgalPolyhedron> IntersectionPrimitive2;
		typedef CGAL::AABB_face_graph_triangle_primitive<CgalPolyhedron> IntersectionPrimitive;
		typedef CGAL::AABB_traits<Kernel, IntersectionPrimitive> Traits;
		typedef CGAL::AABB_traits<Kernel, IntersectionPrimitive2> Traits2;
		typedef CGAL::AABB_tree<Traits> AABBTree;
		typedef CGAL::AABB_tree<Traits2> AABBTree2;
		typedef Kernel::Segment_3 Segment;
		typedef Kernel::Ray_3 Ray;
		typedef boost::optional<AABBTree::Intersection_and_primitive_id<Kernel::Segment_3>::Type > SegmentIntersection;
		typedef boost::optional<AABBTree::Intersection_and_primitive_id<Kernel::Plane_3>::Type> PlaneIntersection;


		//
		// BGL property map which indicates whether an edge is marked as non-removable
		//
		typedef struct Border_is_constrained_edge_map {
			static CGALWrapper::CgalPolyhedron* sm_ptr;
			typedef boost::graph_traits<CGALWrapper::CgalPolyhedron>::edge_descriptor key_type;
			typedef boost::graph_traits<CGALWrapper::CgalPolyhedron>::vertex_descriptor vertex_type;
			typedef bool value_type;
			typedef value_type reference;
			typedef boost::readable_property_map_tag category;
			Border_is_constrained_edge_map() {}
			friend bool get(Border_is_constrained_edge_map m, const key_type& edge) {
				return halfedge(edge, *sm_ptr)->isCrossingEdge;
			}
		};

		

		typedef CGAL::Surface_mesh_simplification::Constrained_placement<
			CGAL::Surface_mesh_simplification::Midpoint_placement<CgalPolyhedron>,
			Border_is_constrained_edge_map > Placement;
	}
}


#endif

