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

#include "Utils/Logger.h"
#include "Utils/Utils.h"

namespace Chimera {

	namespace Core {

		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		Logger::Logger() : m_enabled(true), m_terminalOutput(std::cerr.rdbuf()) {
			time_t rawTime;
			struct tm *timeInfo = new tm();
			
			time(&rawTime);
			localtime_s(timeInfo, &rawTime);
			
			m_logFileStr = "Flow Logs/logFile ";
			m_logFileStr += "Chimera";
			/*m_logFileStr += intToStr(timeInfo->tm_hour) + "h" + intToStr(timeInfo->tm_min) + "m" + " ";
			m_logFileStr += intToStr(timeInfo->tm_mon) + "-" + intToStr(timeInfo->tm_mday) + "-" + intToStr(timeInfo->tm_year - 100);*/
			m_logFileStr += ".log";

			
			
			
			m_allLogFileOutput.open(m_logFileStr.c_str(), ios_base::out);
			m_defaultLogLevel = Log_NormalPriority;
			m_showPriority = Log_NormalPriority;
		}

		Logger::~Logger() {
			m_allLogFileOutput.close();
		}

		Logger & Logger ::operator<<(ostream& (*pFunc)(ostream&)) {
			if (m_enabled) {
				if(m_defaultLogLevel >= m_showPriority )
					m_terminalOutput<< pFunc;

				m_allLogFileOutput << pFunc;
			}
			return *this;
		}

		void Logger::log(const string &logMsg, logLevel_t logLevel) {
			if (m_enabled) {
				if(logLevel >= m_showPriority )
					m_terminalOutput<< logMsg << endl;

				m_allLogFileOutput << logMsg << endl;
			}
		}
	}
}
