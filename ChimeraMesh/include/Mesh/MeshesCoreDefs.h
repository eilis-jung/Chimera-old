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

#ifndef __CHIMERA_MESH_CORE_DEFS_H__
#define __CHIMERA_MESH_CORE_DEFS_H__

#pragma once

namespace Chimera {
	namespace Meshes {

		typedef enum vertexType_t {
			gridVertex,			 //On top of a grid vertex
			edgeVertex,			 //On top of grid edge, but not on top of a vertex
			faceVertex,			 //On top of a grid face, but not on top of an edge
			geometryVertex,		 //On top of the geometry, but not on top of an edge/face
			borderGeometryVertex //On top of the geometry and on a border
		} vertexType_t;


		typedef enum halfEdgeLocation_t {
			rightHalfEdge,
			bottomHalfEdge,
			leftHalfEdge,
			topHalfEdge,
			geometryHalfEdge,
			connectionHalfEdge //Used for connect disconnected regions
		} halfEdgeLocation_t;

		typedef enum edgeType_t {
			yAlignedEdge,
			xAlignedEdge,
			zAlignedEdge,
			geometricEdge
		} edgeType_t;

		typedef enum faceLocation_t {
			XZFace,	//Bottom/Top Faces
			YZFace,	//Left/Right Faces
			XYFace,	//Back/Front Faces
			geometricFace
		} faceLocation_t;

		typedef enum halfFaceLocation_t {
			rightHalfFace,
			bottomHalfFace,
			leftHalfFace,
			topHalfFace,
			frontHalfFace,
			backHalfFace,
			geometryHalfFace
		} halfFaceLocation_t;
	}
}

#endif