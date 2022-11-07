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

#ifndef __CHIMERA_IV_LEVELSET_2D__
#define __CHIMERA_IV_LEVELSET_2D__
#pragma once

#include "LevelSets/LevelSet2D.h"

namespace Chimera {
	namespace LevelSets {

		/** Initial Value Level-Set. It is solved iteratively both by the narrow-band level set method  and traditional approach. */

		class IVLevelSet2D : public LevelSet2D {

		public:

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			IVLevelSet2D(const params_t &params) : LevelSet2D(params) {

			}

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			/** Updates level set values accordingly with polygon points. If polygon is convex
			 ** (params.convexPolygon), then the negative distance is stored for points inside
			 ** the polygon.*/
			void updateLevelSetValues();
			void updateDistanceField(const vector<Vector2> &isocontourPoints, int bandSize);

			
		private:
			
		};
	}
}

#endif
