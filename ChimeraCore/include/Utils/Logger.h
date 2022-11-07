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

#ifndef __FOUNDATION_LOGGER_HPP__
#define __FOUNDATION_LOGGER_HPP__
#pragma once

#include "Config/ChimeraConfig.h"
#include "Utils/Singleton.h"


using namespace std;


namespace Chimera {
	namespace Core {

		/** Log levels: can use to control the application log displayed to the user. */
		typedef enum logLevel_t {
			Log_LowPriority,
			Log_NormalPriority,
			Log_HighPriority
		} logLevel_t;
		
		/** Logger class. */
		class Logger : public Singleton<Logger> {
		public:

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			Logger();
			~Logger();

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			template<class T>
			Logger& operator<<(const T& info);

			Logger& operator<<(ostream& (*pFunc)(ostream&));	//	endl, flush...

			static Logger & get() {
				return *getInstance();
			}

			void log(const string &logMsg, logLevel_t logLevel);

			/************************************************************************/
			/* Access                                                               */
			/************************************************************************/
			void setDefaultLogLevel(logLevel_t logLevel) {
				m_defaultLogLevel = logLevel;
			}

			logLevel_t getDefaultLogLevel() const {
				return m_defaultLogLevel;
			}

			void setShowPriority(logLevel_t showPriority) {
				m_showPriority = showPriority;
			}

			logLevel_t getShowPriority() const {
				return m_showPriority;
			}

		private:

			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			ostream m_terminalOutput;
			string m_logFileStr;
			ofstream m_allLogFileOutput;

			bool m_enabled;
			logLevel_t m_defaultLogLevel;
			logLevel_t m_showPriority;
		};

		template<class T>
		Logger& Logger::operator<<(const T& info) {
			if (m_enabled) {
				if(m_defaultLogLevel >= m_showPriority )
					m_terminalOutput<< info;

				m_allLogFileOutput << info;
			}

			return *this;
		}
	}

		

}


#endif //__LOGGER_HPP__
