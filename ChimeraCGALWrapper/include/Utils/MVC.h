#ifndef __CHIMERA_CGAL_WRAPPER_TRIANGULATOR_H_
#define __CHIMERA_CGAL_WRAPPER_TRIANGULATOR_H_

#pragma once
#include "ChimeraCore.h"
#include "CGALConfig.h"
#include "CGAL/Barycentric_coordinates_2/Mean_value_2.h"
#include "CGAL/Exact_predicates_inexact_constructions_kernel.h"
#include "CGAL/Barycentric_coordinates_2/Generalized_barycentric_coordinates_2.h"
namespace Chimera {

	namespace CGALWrapper {

		using namespace Core;

		typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
		typedef K::Point_2 CGALPoint;
		typedef K::FT  CGALScalar;
		typedef std::pair<double, unsigned> MVCWithInd;

		typedef K::Triangle_2 Triangle2;
		typedef std::vector< std::pair< CGALPoint, CGALScalar > > Point_coordinate_vector;

		typedef std::vector<CGALScalar> CGALScalarVec;
		typedef std::vector<CGALPoint>  CGALPointVec;
		typedef std::back_insert_iterator<CGALScalarVec> Vector_insert_iterator;
		typedef boost::optional<Vector_insert_iterator> Output_type;
		typedef CGAL::Barycentric_coordinates::Mean_value_2<K> Mean_value;
		typedef CGAL::Barycentric_coordinates::Generalized_barycentric_coordinates_2<Mean_value, K> Mean_value_coordinates;

		template <class ScalarType>
		class Triangulator {
		protected:
			CGALPointVecWithInd vertices;
			Delaunay del;
			CGALPointVecWithInd boundaryVerticesWithInd;
			CGALPointVec boundaryVertices;
			int numBoundaryVertices;
		public:
			Triangulator();
			Triangulator(const vector<std::pair<ScalarType, unsigned> > v);
			~Triangulator();
			void getBoundaryVerticesList();
			vector<MVCWithInd> getBarycentricCoordinates(ScalarType v);
		};
	}
}
#endif
