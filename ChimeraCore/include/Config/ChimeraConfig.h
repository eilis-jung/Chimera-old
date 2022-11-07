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

#ifndef __M_CHIMERA_CFG__H_
#define __M_CHIMERA_CFG__H_
#pragma once

/** Selects between dynamic and static linking:*/
#ifdef CHIMERA_BUILD_DLL
#define CHIMERA_API __declspec(dllexport)
#elif defined(MATH_USE_DLL)
#define CHIMERA_API __declspec(dllimport)
#else //Static linking
#define CHIMERA_API
#endif


/************************************************************************/
/* Windows DEFS                                                         */
/************************************************************************/
#if defined _WIN32
	/** Inline definition */
	#if defined(__MINGW32__) || defined(__CYGWIN__) || (defined (_MSC_VER) && _MSC_VER < 1300)
		#define FORCE_INLINE inline
	#else
		#pragma warning(disable: 4324) // disable padding warning
		#pragma warning(disable:4786) // Disable the "debug name too long" warning

		#ifdef __CUDACC__
		#define FORCE_INLINE __forceinline__
		#else
		#define FORCE_INLINE __forceinline
		#endif

	#endif
	
	/**Pragma disables */
	#pragma warning(disable: 4514) //Removes the inline warnings 'unreferenced inline function has been removed'
#else

//TODO: inline for other plataforms or OS

#endif

/************************************************************************/
/* STL                                                                  */
/************************************************************************/
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <random>
#include <typeinfo>

/** Containers */
#include <vector>
#include <map>
/**TR1: */
#include <unordered_map>

/** Math */
#include <complex>

/************************************************************************/
/* Cplusplus utils                                                      */
/************************************************************************/
#include <cassert>
#include <cfloat>
#include <climits>
#include <limits>
#include <cmath>
#include <math.h>
#include <string>
#include <exception>
#include <sstream>
#include <memory>
#include <iostream>
#include <ctime>
#include <fstream>
#include <queue>
#include <functional>
#include <algorithm>
#include <type_traits>


typedef unsigned int uint;

#endif