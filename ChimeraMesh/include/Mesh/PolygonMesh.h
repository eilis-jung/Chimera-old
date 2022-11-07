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

#ifndef __CHIMERA_POLYGON_MESH_H_
#define __CHIMERA_POLYGON_MESH_H_

#pragma once

#include "ChimeraCore.h"
#include "ChimeraMath.h"
#include "Grids/CutCells3D.h"

namespace Chimera {
	using namespace Math;
	namespace Data {

		class PolygonMesh {
		public:

			typedef struct polygon_t {
				vector<int> pointsIndexes;
			};

			typedef enum nodeType_t {
				gridNode,
				geometryNode,
				centroidNode
			} nodeType_t;

			PolygonMesh(CutCells2D *pCutCells);

			const vector<Vector2> & getPoints() const {
				return m_points;
			}

			const vector<polygon_t> & getPolygons() const {
				return m_polygons;
			}

			const vector<nodeType_t> & getNodeTypes() const {
				return m_nodeTypes;
			}

			bool isInsideMesh(const Vector2 & position);

		private:
#pragma region ComparisionFunctions
			static bool comparePairs(pair<Vector2, int *> a, pair<Vector2, int *> b);
			static bool comparePairs_b(pair<Vector2, int *> a, pair<Vector2, int *> b);
			static bool uniqueVectors(Vector2 a, Vector2 b);
#pragma endregion

			typedef struct polygonPointers_t {
				vector<int *> pointsIndexes;
			};

			vector<Vector2> m_points;
			vector<polygon_t> m_polygons;
			vector<nodeType_t> m_nodeTypes;
		};
	}

}
#endif