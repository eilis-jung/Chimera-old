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

#ifndef _CHIMERA_CUSP_CONFIG
#define _CHIMERA_CUSP_CONFIG

#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
#endif

#include "CudaConfig.h"

#if defined(__CUDACC__) || defined(__FAUX_CUDACC__) 
	#ifndef _CHIMERA_MATH_CUSP_CONFIG_CU_INCLUDED_
	#define _CHIMERA_MATH_CUSP_CONFIG_CU_INCLUDED_
		
		/** Base */
		#include <cusp/print.h>
		#include <cusp/dia_matrix.h>
		#include <cusp/coo_matrix.h>
		#include <cusp/ell_matrix.h>
		#include "Math/Matrix/Matrix3x3.h"
		#ifdef __CUDACC__
		#include <cusp/monitor.h>
			/** Preconditioners */
			#include <cusp/precond/diagonal.h>
			#include <cusp/precond/ainv.h>
			#include <cusp/precond/aggregation/smoothed_aggregation.h>
			/** Solvers */
			#include <cusp/krylov/cg.h>
			#include <cusp/csr_matrix.h>
			#include <cusp/print.h>
			#include <cusp/multiply.h>

		#endif
		typedef struct matrix33 {
			float3 column[3];
			#ifdef __CUDACC__
				__device__ float3 operator [] (int i) { return column[i]; }

				void operator=(Chimera::Core::Matrix3x3 mat) { 
					for(int i = 0; i < 3; i++) {
						column[i].x = mat[i].x;
						column[i].y = mat[i].y;
						column[i].z = mat[i].z;
					}
			
				}
			#endif
		} matrix33;
	#endif
#else

	namespace thrust {
		struct device_space_tag {};
		struct host_space_tag {};
		struct any_space_tag {};

		#define THRUST_DEVICE_BACKEND_CUDA    1
		#define THRUST_DEVICE_BACKEND_OMP     2

		#define THRUST_DEVICE_BACKEND THRUST_DEVICE_BACKEND_CUDA
		namespace detail
		{

			// define these in detail for now
			struct cuda_device_space_tag : device_space_tag {};
			struct omp_device_space_tag : device_space_tag {};

			#if   THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_CUDA
						typedef cuda_device_space_tag default_device_space_tag;
			#elif THRUST_DEVICE_BACKEND == THRUST_DEVICE_BACKEND_OMP
						typedef omp_device_space_tag  default_device_space_tag;
			#else
			#error Unknown device backend.
			#endif // THRUST_DEVICE_BACKEND

		} // end namespace detail


	}
	
	// DO NOT USE MATRIX3x3 ON HOST CODE
	struct matrix33 {
	};

	namespace cusp {
		template<typename T, typename MemorySpace>
		class identity_operator;

		//Forward verbose_monitor definition
		template<typename T>
		class verbose_monitor;

		//Forward array definition
		template<typename T, typename MemorySpace>
		class array1d;

		//Forward dia_matrix definition
		template<typename R, typename T, typename MemorySpace>
		class dia_matrix;

		//Forward ell_matrix definition
		template<typename R, typename T, typename MemorySpace>
		class ell_matrix;

		//Forward coo_matrix definition
		template<typename R, typename T, typename MemorySpace>
		class coo_matrix;

		//Forward coo_matrix definition
		template<typename R, typename T, typename MemorySpace>
		class hyb_matrix;

		//Monitor declaration
		template<typename T>
		class default_monitor;

		////Work around for redefining typedef structs :)
		typedef thrust::detail::default_device_space_tag	device_memory;
		typedef thrust::host_space_tag						host_memory;
		typedef thrust::any_space_tag						any_memory;

		template<typename T, typename MemorySpace>
		struct default_memory_allocator;

		template <typename MemorySpace1, typename MemorySpace2=any_memory, typename MemorySpace3=any_memory>
		struct minimum_space;

		namespace precond {
			//Forward preconditioners
			template<typename R, typename T>
			class diagonal;

			template<typename R, typename T>
			class bridson_ainv;

			namespace aggregation {
				template<typename R, typename S, typename T>
				class smoothed_aggregation;
			}
		}
	}
	#endif

	

#endif

