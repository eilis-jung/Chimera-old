#include "Utils/Utils.h"
#include "Utils/ConversionManager.h"
#include <CGAL/Aff_transformation_3.h>

namespace Chimera {
	namespace CGALWrapper {

		void triangulatePolyhedron(CgalPolyhedron *pCGALPoly)  {
			CGAL::triangulate_polyhedron(*pCGALPoly);
			for (CgalPolyhedron::Facet_iterator it = pCGALPoly->facets_begin(); it != pCGALPoly->facets_end(); ++it) {
				CgalPolyhedron::Facet_handle fit = it;
				CgalPolyhedron::Traits::Vector_3 normal = compute_facet_normal<CgalPolyhedron::Facet, CgalPolyhedron::Traits>(*fit);
				fit->normal = Conversion::vec3ToVec<Vector3D>(normal);
			}

			//Fixing vertices indexes
			int tempIndex = 0;
			for (CgalPolyhedron::Vertex_iterator it = pCGALPoly->vertices_begin(); it != pCGALPoly->vertices_end(); it++) {
				CgalPolyhedron::Vertex_handle vh(it);
				vh->id = tempIndex++;
			}
		}


		void translatePolygon(CgalPolyhedron *pPoly, const Vector3D &translationVec) {
			Kernel::Vector_3 cgalVec = Conversion::vecToVec3(translationVec);
			CGAL::Aff_transformation_3<Kernel> translationMat(CGAL::Translation(), cgalVec);
			std::transform(pPoly->points_begin(), pPoly->points_end(), pPoly->points_begin(), translationMat);
		}

		void rotatePolygonZ(CgalPolyhedron *pPoly, DoubleScalar rotationAngle) {
			CGAL::Aff_transformation_3<Kernel> rotationMat(	cos(rotationAngle), -sin(rotationAngle), 0.0,
															sin(rotationAngle), cos(rotationAngle), 0.0,
															0.0, 0.0, 1.0);
			std::transform(pPoly->points_begin(), pPoly->points_end(), pPoly->points_begin(), rotationMat);
		}
	}
}