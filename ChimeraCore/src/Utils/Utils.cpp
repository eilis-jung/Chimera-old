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

#include "Utils/Utils.h"

namespace Chimera {
	namespace Core{

		string intToStr(int cInt) {
			os.str("");
			os << cInt;
			return(os.str());
		}

		string scalarToStr(float cScalar) {
			os.str("");
			os << cScalar;
			return(os.str());
		}

		string scalarToStr(double cScalar){
			os.str("");
			os << cScalar;
			return(os.str());
		}

		void exitProgram() {
			cout << "Press any key to exit the program. " << endl;
			char c;
			cin >> c;
			exit(0);
		}


		void exitProgram(const string &errorMsg) {
			Logger::getInstance()->log(errorMsg, Log_HighPriority);
			exitProgram();
		}

		unsigned int getTickValue() {
#ifdef _WIN32
			static LARGE_INTEGER yo;
			static LONGLONG counts;
#else
			static timeval tp, initialTime;
#endif
			static bool initialized = false;
			if (initialized == false) {
#ifdef _WIN32
				QueryPerformanceFrequency(&yo);
				counts = yo.QuadPart / 1000;
#else
				gettimeofday(&initialTime, 0);
#endif
				initialized = true;
			}

#ifdef _WIN32
			LARGE_INTEGER PerfVal;
			QueryPerformanceCounter(&PerfVal);
			return (unsigned int) (PerfVal.QuadPart / counts);
#else
			gettimeofday(&tp, 0);
			// Seconds to ms and microseconds to ms
			return ( tp.tv_sec - initialTime.tv_sec) * 1000 + (tp.tv_usec - initialTime.tv_usec) / 1000;
#endif
		}

	}
}