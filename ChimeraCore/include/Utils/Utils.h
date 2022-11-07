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
//	
#ifndef _UTILS_FUNCTIONS_H__
#define _UTILS_FUNCTIONS_H__


#include "Config/ChimeraConfig.h"
#include "Utils/Logger.h"
#include "Utils/Times.h"

using namespace std;

namespace Chimera {

	namespace Core {

		static stringstream os;

		/************************************************************************/
		/* General                                                              */
		/************************************************************************/
		string intToStr(int cInt);

		string scalarToStr(float cScalar);

		string scalarToStr(double cScalar);

		FORCE_INLINE int randomNumber(int maxNum) {
			return(rand() % maxNum);
		}

		FORCE_INLINE void safeDelete(void * dMem) {
			if(dMem != NULL) {
				delete dMem;
				dMem = NULL;
			}
		}

		template <class typeCast>
		static FORCE_INLINE void * voidCast(const typeCast &castedClass) {
			return (void *) &castedClass;
		}

		void exitProgram();
		void exitProgram(const string &errorMsg);
		unsigned int getTickValue();

	}
}
#endif