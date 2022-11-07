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

//////////////////////////////////////////////////////////////////////
// Took from:
//////////////////////////////////////////////////////////////////////
/*!
\file       singleton.h
\brief      Implementation of the CSingleton template class.
\author     Brian van der Beek
*/

#ifndef __SINGLETON_H__
#define __SINGLETON_H__

//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////

namespace Chimera {
	namespace Core {


		//////////////////////////////////////////////////////////////////////
		//
		//////////////////////////////////////////////////////////////////////

		//! The CSingleton class is a template class for creating singleton objects.

		/*!
		When the static Instance() method is called for the first time, the singleton 
		object is created. Every sequential call returns a reference to this instance.
		The class instance can be destroyed by calling the DestroyInstance() method.
		*/
		template <typename T>
		class Singleton {
		public:

			//! Gets a reference to the instance of the singleton class.

			/*!
			\return A reference to the instance of the singleton class.
			If there is no instance of the class yet, one will be created.
			*/
			static T* getInstance() {
				if (m_instance == NULL) m_instance = new T;

				return m_instance;
			};

			//! Destroys the singleton class instance.

			/*!
			Be aware that all references to the single class instance will be
			invalid after this method has been executed!
			*/
			static void destroyInstance() {
				delete m_instance;
				m_instance = NULL;
			};

		protected:

			// shield the constructor and destructor to prevent outside sources
			// from creating or destroying a CSingleton instance.

			//! Default constructor.

			Singleton() {
			};


			//! Destructor.

			virtual ~Singleton() {
			};

		protected:

			//! Copy constructor.

			Singleton(const Singleton& source) {
			};

			static T* m_instance; //!< singleton class instance
		};

		//! static class member initialisation.
		template <typename T> T* Singleton<T>::m_instance = NULL;


		//////////////////////////////////////////////////////////////////////
		//
		//////////////////////////////////////////////////////////////////////

	}
}
//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////

#endif // ! defined __SINGLETON_H__
