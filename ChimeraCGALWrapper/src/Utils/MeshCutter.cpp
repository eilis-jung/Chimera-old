#include "Utils/MeshCutter.h"
#include "Utils/ConversionManager.h"
#include "Utils/Utils.h"
#include "Mesh/MeshUtils.h"

namespace Chimera {
	namespace CGALWrapper{

		#pragma region Constructors
		MeshCutter::MeshCutter() {

		}
		#pragma endregion

		#pragma region ComparisonFunctions
		bool MeshCutter::comparePairs(pair<CgalPolyhedron::Point_3, unsigned int *> a, pair<CgalPolyhedron::Point_3, unsigned int *> b) {
			if (a.first.x() != b.first.x()) {
				return a.first.x() < b.first.x();
			}
			else if (a.first.y() != b.first.y()) {
				return a.first.y() < b.first.y();
			}
			else if (a.first.z() != b.first.z()) {
				return a.first.z() < b.first.z();
			}
			else {
				return *a.second < *b.second;
			}
		}
		bool MeshCutter::comparePairs2(pair<CgalPolyhedron::Point_3, unsigned int *> a, pair<CgalPolyhedron::Point_3, unsigned int *> b) {
			return *a.second < *b.second;
		}

		bool MeshCutter::comparePairs3(pair<Vector3, Vector3> a, pair<Vector3, Vector3> b) {
			return a.first < b.first;
		}

		bool MeshCutter::comparePairsInts(unsigned int * a, unsigned int * b) {
			return *a < *b;
		}

		#pragma endregion

		#pragma region Functionalities
		void MeshCutter::halfspaceIntersection(CgalPolyhedron *pCgalPoly, const Data::PlaneMesh &plane) {
			Kernel::Plane_3 cgalPlane = Conversion::planeMeshToPlane3(plane);
			AABBTree tree(faces(*pCgalPoly).first, faces(*pCgalPoly).second, *pCgalPoly);

			/** Compute first intersection */
			list<PlaneIntersection> intersections;
			auto anyIntersection = tree.any_intersection(cgalPlane);
			if (anyIntersection) {
				CgalPolyhedron::Face_handle faceHandle = anyIntersection->second;
				CgalPolyhedron::Halfedge_handle edgeHandle = faceHandle->halfedge();

				bool v1Check = cgalPlane.has_on_positive_side(edgeHandle->vertex()->point());
				bool v2Check = cgalPlane.has_on_positive_side(edgeHandle->opposite()->vertex()->point());

				/** We need vertices from opposite sides from the plane: */
				while ((v1Check && v2Check) || (!v1Check && !v1Check)) {
					edgeHandle = edgeHandle->next();
					v1Check = cgalPlane.has_on_positive_side(edgeHandle->vertex()->point());
					v2Check = cgalPlane.has_on_positive_side(edgeHandle->opposite()->vertex()->point());
				}
				if (v1Check)
					edgeHandle = polyhedronCutPlane(*pCgalPoly, edgeHandle->opposite(), cgalPlane);
				else
					edgeHandle = polyhedronCutPlane(*pCgalPoly, edgeHandle, cgalPlane);

			}
		}

		pair<vector<CgalPolyhedron::Point_3>, vector<vector<unsigned int>>> MeshCutter::fixPolygon(CgalPolyhedron *pCgalPoly) {
			pair<vector<CgalPolyhedron::Point_3>, vector<vector<unsigned int>>> correctedPolygon;

			int tempIndex = 0;
			vector<CgalPolyhedron::Point_3> tempVertices;
			for (auto it = pCgalPoly->vertices_begin(); it != pCgalPoly->vertices_end(); it++) {
				CgalPolyhedron::Vertex_handle vh(it);
				vh->id = tempIndex++;
				tempVertices.push_back(it->point());
			}

			vector<pair<CgalPolyhedron::Point_3, unsigned int *>> allPairs;
			for (CgalPolyhedron::Facet_iterator it = pCgalPoly->facets_begin(); it != pCgalPoly->facets_end(); ++it) {
				auto hfc = it->facet_begin();

				for (int j = 0; j < it->size(); ++j, ++hfc) {
					CgalPolyhedron::Vertex_handle vh(hfc->vertex());
					//pair<CgalPolyhedron::Point_3, unsigned int *> tempPair(hfc->vertex()->point(), &vh->id);
					allPairs.push_back(pair<CgalPolyhedron::Point_3, unsigned int *>(hfc->vertex()->point(), &vh->id));
				}
			}

			sort(allPairs.begin(), allPairs.end(), MeshCutter::comparePairs);
			//sort(allPairs.begin(), allPairs.end(), MeshCutter::comparePairs2);

			for (int i = 0; i < allPairs.size(); i++) {
				int nextIndex = roundClamp<int>(i + 1, 0, allPairs.size());
				while (allPairs[i].first == allPairs[nextIndex].first) {
					if (*allPairs[nextIndex].second > *allPairs[i].second) {
						*allPairs[nextIndex].second = *allPairs[i].second;
					}
					else if (*allPairs[nextIndex].second < *allPairs[i].second){
						*allPairs[i].second = *allPairs[nextIndex].second;
					}
					i++;
					if (i >= allPairs.size())
						break;
					nextIndex = roundClamp<int>(i + 1, 0, allPairs.size());
				}
			}

			for (int i = 0; i < allPairs.size();) {
				int nextIndex = roundClamp<int>(i + 1, 0, allPairs.size());
				if (correctedPolygon.first.size() > 0 && correctedPolygon.first.back() != tempVertices[*allPairs[i].second]) {
					correctedPolygon.first.push_back(tempVertices[*allPairs[i].second]);
				}
				else {
					correctedPolygon.first.push_back(tempVertices[*allPairs[i].second]);
				}
				
				if (*allPairs[i].second == *allPairs[nextIndex].second) {
					while (allPairs[i].first == allPairs[nextIndex].first) {
						*allPairs[i].second = correctedPolygon.first.size() - 1;
						i++;
						nextIndex = roundClamp<int>(i + 1, 0, allPairs.size());
					}
					*allPairs[i].second = correctedPolygon.first.size() - 1;
					i++;
				}
				else {
					i++;
				}
			}

			for (CgalPolyhedron::Facet_iterator it = pCgalPoly->facets_begin(); it != pCgalPoly->facets_end(); ++it) {
				auto hfc = it->facet_begin();

				vector<unsigned int> currFace;
				for (int j = 0; j < it->size(); ++j, ++hfc) {
					CgalPolyhedron::Vertex_handle vh(hfc->vertex());
					currFace.push_back(vh->id);
				}
				vector<unsigned int *> tempCurrFace;// = currFace;
				for (int j = 0; j < currFace.size(); j++) {
					tempCurrFace.push_back(&currFace[j]);
				}
				sort(tempCurrFace.begin(), tempCurrFace.end(), comparePairsInts);
				bool validFace = true;
				for (int j = 0; j < tempCurrFace.size() - 1; j++) {
					if (*tempCurrFace[j] == *tempCurrFace[j + 1]) {
						*tempCurrFace[j] = UINT_MAX;
						//validFace = false;
						//break;
					}
				}
				for (int j = 0; j < currFace.size(); ) {
					if (currFace[j] == UINT_MAX)
						currFace.erase(currFace.begin() + j);
					else
						j++;
				}
				if (currFace.size() > 2)
					correctedPolygon.second.push_back(currFace);
			}

			return correctedPolygon;
		}

		vector<Vector3D> MeshCutter::planeIntersection(CgalPolyhedron *pCgalPoly, const Data::PlaneMesh &plane) {
			Kernel::Plane_3 cgalPlane = Conversion::planeMeshToPlane3(plane);
			AABBTree tree(faces(*pCgalPoly).first, faces(*pCgalPoly).second, *pCgalPoly);

			std::vector<pair<Vector3D, Vector3D>> pointsPairs;
			std::vector<Vector3D> points;
			std::vector<PlaneIntersection> intersections;
			auto anyIntersection = tree.any_intersection(cgalPlane);
			if (anyIntersection) {
				tree.all_intersections(cgalPlane, std::back_inserter(intersections));
				for (std::vector<PlaneIntersection>::iterator it = intersections.begin(), end = intersections.end(); it != end; ++it) {
					Segment *pCurrSegment = boost::get<Segment>(&(it->value().first));
					if (pCurrSegment) {
						pointsPairs.push_back(pair<Vector3D, Vector3D>(Conversion::pointToVec<Vector3D>(pCurrSegment->source()), Conversion::pointToVec<Vector3D>(pCurrSegment->target())));
						points.push_back(Conversion::pointToVec<Vector3D>(pCurrSegment->source()));
						points.push_back(Conversion::pointToVec<Vector3D>(pCurrSegment->target()));
						/*points.push_back();*/
						//points.push_back(Conversion::point3ToVec(it->value().second->halfedge()->vertex()->point()));
						/*if (points.size() > 0) {
							Vector3 tempPoint = Conversion::point3ToVec(pCurrSegment->source());
							if (points.back() == tempPoint) {
								tempPoint = Conversion::point3ToVec(pCurrSegment->target());
							}
							points.push_back(tempPoint);
						}
						else {
							points.push_back(Conversion::point3ToVec(pCurrSegment->source()));
						}*/
					}	
				}
				/*sort(pointsPairs.begin(), pointsPairs.end(), comparePairs3);
				pair<Vector3, Vector3> currPair;
				for (int i = 0; i < pointsPairs.size(); i++) {
					if (points.size() == 0) {
						currPair = pointsPairs.front();
					}
					else {
						pair<Vector3, Vector3> invertedPair(currPair.second, currPair.first);
						int currIndex = lower_bound(pointsPairs.begin(), pointsPairs.end(), invertedPair, comparePairs3) - pointsPairs.begin();
						currPair = pointsPairs[currIndex];
					}
					points.push_back(currPair.first);
				}*/
			}
			return points;	
		}

		vector<vector<Vector3D>> MeshCutter::polygonSlicer(CgalPolyhedron *pCgalPoly, const Vector3 &normal, const Vector3 &origin, const Vector3 &increment, int numSlices) {
			//AABBTree tree(faces(*pCgalPoly).first, faces(*pCgalPoly).second, *pCgalPoly);

			typedef std::vector<Kernel::Point_3> Polyline;
			typedef std::list< Polyline > Polylines;
			

			Polylines polylines;
			Kernel::Vector_3 planeNormal = Conversion::vecToVec3(convertToVector3D(normal));
			for (int i = 0; i < numSlices; i++) {
				Kernel::Point_3 planeOrigin = Conversion::vecToPoint3(convertToVector3D(origin + increment*i));
				Kernel::Plane_3 intersectionPlane(planeOrigin, planeNormal);
				//auto anyIntersection = tree.any_intersection(intersectionPlane);
				//if (anyIntersection) {
				//if (tree.do_intersect(intersectionPlane)) {
				{	
					CGAL::Polygon_mesh_slicer_3<CgalPolyhedron, Kernel> slicer(*pCgalPoly);
					slicer(intersectionPlane, std::back_inserter(polylines));
				}
				/*if (polylines.size()) {
					if (polylines.back().size() > 2) {
						if (polylines.back().front() == polylines.back().back()) {
							polylines.back().pop_back();
						}
					}
				}*/
				
			}			
			
			vector<vector<Vector3D>> finalPointsVec;
			BOOST_FOREACH(Polyline pl, polylines){
				vector<Vector3D> pointsVec;
				for (int i = 0; i < pl.size(); i++) {
					pointsVec.push_back(Conversion::pointToVec<Vector3D>(pl[i]));
				}
				//reverse(pointsVec.begin(), pointsVec.end());
				finalPointsVec.push_back(pointsVec);
			}
			return finalPointsVec;
		}

		vector<vector<Vector3D>> MeshCutter::polygonSlicer(const vector<CgalPolyhedron *> &pCgalPolys, const Vector3 &normal, const Vector3 &origin, const Vector3 &increment, int numSlices) {
			typedef std::vector<Kernel::Point_3> Polyline;
			typedef std::list< Polyline > Polylines;


			Polylines polylines;
			Kernel::Vector_3 planeNormal = Conversion::vecToVec3(convertToVector3D(normal));
			for (int i = 0; i < numSlices; i++) {
				Kernel::Point_3 planeOrigin = Conversion::vecToPoint3(convertToVector3D(origin + increment*i));
				Kernel::Plane_3 intersectionPlane(planeOrigin, planeNormal);
				for (int j = 0; j < pCgalPolys.size(); j++) { 
					CGAL::Polygon_mesh_slicer_3<CgalPolyhedron, Kernel> slicer(*pCgalPolys[j]);
					slicer(intersectionPlane, std::back_inserter(polylines));
				}
			}

			vector<vector<Vector3D>> finalPointsVec;
			BOOST_FOREACH(Polyline pl, polylines){
				vector<Vector3D> pointsVec;
				for (int i = 0; i < pl.size(); i++) {
					pointsVec.push_back(Conversion::pointToVec<Vector3D>(pl[i]));
				}
				//reverse(pointsVec.begin(), pointsVec.end());
				finalPointsVec.push_back(pointsVec);
			}
			return finalPointsVec;
		}
		#pragma endregion
	}
}