#include "ChimeraCGALWrapper.h"

namespace Chimera {

	namespace CGALWrapper {
		bool sortCoordinateVec(MVCWithInd a, MVCWithInd b)
		{
			return a.second <= b.second;
		}

		template <class ScalarType>
		Triangulator<ScalarType>::Triangulator() {
		}

		template <class ScalarType>
		Triangulator<ScalarType>::Triangulator(vector<std::pair<ScalarType, unsigned> > v) {
			CGALPointVecWithInd points;
			vector<std::pair<ScalarType, unsigned> >::iterator it;

			numBoundaryVertices = 0;

			for (it = v.begin(); it != v.end(); ++it) {
				std::pair<CGALPoint, unsigned> tempPoint(CGALPoint((*it).first.x, (*it).first.y), (*it).second);
				points.push_back(tempPoint);
			}

			del.insert(points.begin(), points.end());
		}
		template <class ScalarType>
		Triangulator<ScalarType>::~Triangulator() {
		}

		template <class ScalarType>
		void Triangulator<ScalarType>::getBoundaryVerticesList() {
			Vertex_circulator vc = del.incident_vertices(del.infinite_vertex()),
				done(vc);

			if (vc != 0) {
				do {
					boundaryVerticesWithInd.push_back(CGALPointWithInd(vc->point(), vc->info()));
					boundaryVertices.push_back(vc->point());
				} while (++vc != done);
			}
			numBoundaryVertices = boundaryVertices.size();
			return;
		}

		template <class ScalarType>
		vector<MVCWithInd> Triangulator<ScalarType>::getBarycentricCoordinates(ScalarType v) {
			const CGAL::Barycentric_coordinates::Type_of_algorithm algorithmType = CGAL::Barycentric_coordinates::FAST;
			const CGAL::Barycentric_coordinates::Query_point_location queryPointLocationType = CGAL::Barycentric_coordinates::ON_BOUNDED_SIDE;

			CGALScalarVec coordinates;
			vector<MVCWithInd> res;

			for (FaceIterator faceIt = del.finite_faces_begin(); faceIt != del.finite_faces_end(); ++faceIt)
			{
				Delaunay::Face_handle face = faceIt;
				Triangle2 tri(face->vertex(0)->point(), face->vertex(1)->point(), face->vertex(2)->point());

				if (tri.has_on_positive_side(CGALPoint(v.x, v.y)) || tri.has_on_boundary(CGALPoint(v.x, v.y)))
				{
					CGALPointVec triPoints;
					triPoints.push_back(face->vertex(0)->point());
					triPoints.push_back(face->vertex(1)->point());
					triPoints.push_back(face->vertex(2)->point());

					Mean_value_coordinates mvc(triPoints.begin(), triPoints.end());

					const Output_type mvcRes = mvc(CGALPoint(v.x, v.y), std::back_inserter(coordinates), queryPointLocationType, algorithmType);

					res.push_back(MVCWithInd(coordinates[0], face->vertex(0)->info()));
					res.push_back(MVCWithInd(coordinates[1], face->vertex(1)->info()));
					res.push_back(MVCWithInd(coordinates[2], face->vertex(2)->info()));
					break;
				}
			}

			/*vector<MVCWithInd>::iterator it;

			for (it = res.begin(); it != res.end(); it++)
			{
			cout << "Coordinate " << (*it).second << " = " << (*it).first << endl;
			}*/
			return res;
		}
		template class Triangulator <Vector2>;
		template class Triangulator <Vector2D>;

	}
}



int test()
{
	using namespace Chimera;
	using Chimera::Core::Vector2;
	using namespace std;

	vector<std::pair<Core::Vector2, unsigned> > t;
	t.push_back(std::make_pair(Vector2(0, 0), 0));
	t.push_back(std::make_pair(Vector2(0, 3), 1));
	t.push_back(std::make_pair(Vector2(3, 0), 2));
	t.push_back(std::make_pair(Vector2(3, 3), 3));
	t.push_back(std::make_pair(Vector2(1, 1), 4));
	t.push_back(std::make_pair(Vector2(1, 2), 5));
	t.push_back(std::make_pair(Vector2(2, 1), 6));
	t.push_back(std::make_pair(Vector2(2, 2), 7));
	t.push_back(std::make_pair(Vector2(1.5, 4), 8));
	CGALWrapper::Triangulator<Vector2> tri(t);
	tri.getBoundaryVerticesList();
	tri.getBarycentricCoordinates(Vector2(0.3, 0.4));
	return 0;
}
