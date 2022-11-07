#pragma once

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


#include "Config/ChimeraConfig.h"

namespace Chimera {
	namespace Core {


		template <class... Ts> struct tuple {};

		template <class T, class... Ts>
		struct tuple<T, Ts...> : tuple<Ts...> {
			tuple(T t, Ts... ts) : tuple<Ts...>(ts...), tail(t) {}

			map<string, T> tail;
		};

		template <size_t, class> struct elem_type_holder;

		template <class T, class... Ts>
		struct elem_type_holder<0, tuple<T, Ts...>> {
			typedef T type;
		};

		template <size_t k, class T, class... Ts>
		struct elem_type_holder<k, tuple<T, Ts...>> {
			typedef typename elem_type_holder<k - 1, tuple<Ts...>>::type type;
		};

		template <size_t k, class... Ts>
		typename std::enable_if<
			k == 0, typename elem_type_holder<0, tuple<Ts...>>::type&>::type
			get(tuple<Ts...>& t) {
			return t.tail;
		}

		template <size_t k, class T, class... Ts>
		typename std::enable_if<
			k != 0, typename elem_type_holder<k, tuple<T, Ts...>>::type&>::type
			get(tuple<T, Ts...>& t) {
			tuple<Ts...>& base = t;
			return get<k - 1>(base);
		}


		template<class ... Types> struct VariadicAttributes {
			
			tuple<Types> m_tuples;

			typename elem_type_holder<1, tuple<double, int, const char*>>::type foo;
			std::cout << typeid(foo).name() << "\n";
		};

	}
}
