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

#include "Math/Matrix/Quaternion.h"


namespace Chimera {

	namespace Core {

		//////////////////////////////////////////////////////////////////////
		// Construction/Destruction
		//////////////////////////////////////////////////////////////////////

		Quaternion::~Quaternion()
		{

		}

		//////////////////////////////////////////////////////////////////////
		//
		//////////////////////////////////////////////////////////////////////

		void Quaternion::rotate(Vector3 *v)
		{
			//// nVidia SDK implementation
			//Vector3 qvec(x, y, z);
			//Vector3 uv(qvec.cross(*v));
			//Vector3 uuv(qvec.cross(uv));
			//uv  *= (2.0f * w);
			//uuv *= 2.0f;
			//Vector3 vtmp = uv + uuv;
			//*v = *v + vtmp;


			Quaternion myInverse;
			myInverse.x = -x;
			myInverse.y = -y;
			myInverse.z = -z;
			myInverse.w = w;

			Quaternion left;
			left.multiply(*this, *v);

			v->x = left.w * myInverse.x + myInverse.w * left.x + left.y * myInverse.z - myInverse.y * left.z;
			v->y = left.w * myInverse.y + myInverse.w * left.y + left.z * myInverse.x - myInverse.z * left.x;
			v->z = left.w * myInverse.z + myInverse.w * left.z + left.x * myInverse.y - myInverse.x * left.y;


		}


	}
}