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

#ifndef _MATH_CONJUGATE_GRADIENT_
#define _MATH_CONJUGATE_GRADIENT_
#pragma  once

#include "ChimeraCore.h"

/************************************************************************/
/* Math API                                                             */
/************************************************************************/
#include "Poisson/PoissonMatrix.h"
#include "Poisson/PoissonSolver.h"

using namespace std;
namespace Chimera {
	namespace Poisson {

	class ConjugateGradient : public PoissonSolver {

	public:
		#pragma region InternalStructures
		typedef enum LSPreconditioner {
			Diagonal,
			AINV,
			SmoothedAggregation,
			NoPreconditioner
		} LSPreconditioner;

	private:
		
		/** Cuda variables structure */
		class CudaScalarField {
			public:

			cusp::array1d<Scalar, cusp::device_memory> *pDevicePressure;
			cusp::array1d<Scalar, cusp::device_memory> *pDeviceFlux;

			cusp::array1d<Scalar, cusp::host_memory> *pPressure;
			cusp::array1d<Scalar, cusp::host_memory> *pFlux;
			
			
			CudaScalarField(const PoissonSolver::params_t &solverParams, const dimensions_t &gridDimensions);
		};
		#pragma endregion
		
		#pragma region ClassMembers
		CudaScalarField *m_pCudaScalarField;

		/** Monitor properties */
		bool m_dumpToFile;
		bool m_verbose;
		bool m_isSetup;
		cusp::default_monitor<float> *m_pMonitor;

		/** Preconditioners */
		LSPreconditioner m_preconditioner;
		/** Diagonal */
		cusp::precond::diagonal<Scalar, cusp::host_memory> *m_pDiagonalPrecond;
		cusp::precond::diagonal<Scalar, cusp::device_memory> *m_pDeviceDiagonalPrecond;
		/** Bridsons AINV */
		cusp::precond::bridson_ainv<Scalar, cusp::host_memory> *m_pAinvPrecond;
		cusp::precond::bridson_ainv<Scalar, cusp::device_memory> *m_pDeviceAinvPrecond;
		/** Smoothed AMG */
		cusp::precond::aggregation::smoothed_aggregation<Integer, Scalar, cusp::host_memory> *m_pSmoothedPrecond;
		cusp::precond::aggregation::smoothed_aggregation<Integer, Scalar, cusp::device_memory> *m_pDeviceSmoothedPrecond;
		#pragma endregion
		

		#pragma region PrivateFunctionalities
		/** .cu side functions */
		void initializePreconditioners();
		void dumpToFile() const;

		void copyToCudaScalarField(const Array2D<Scalar> *pRhs, bool copyToGPU = true);
		void copyToCudaScalarField(const Array3D<Scalar> *pRhs, bool copyToGPU = true);
		void copyFromCudaScalarField(Array2D<Scalar> *pResult, bool copyFromGpu = true);
		void copyFromCudaScalarField(Array3D<Scalar> *pResult, bool copyFromGpu = true);
		#pragma endregion

	public:
		
		#pragma region Constructors
		ConjugateGradient(const params_t &params, PoissonMatrix *A, LSPreconditioner preconditioner = NoPreconditioner);

		#pragma region Solving
		bool solveCPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult);
		bool solveGPU(const Array<Scalar> *pRhs, Array<Scalar> *pResult);

		/************************************************************************/
		/* Reinitialization                                                     */
		/************************************************************************/
		void resizeScalarFields();
		void reinitializePreconditioners();
	};
}

}
#endif