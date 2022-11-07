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

#ifndef _MATH_SEMI_LAGRANGIAN_H_
#define _MATH_SEMI_LAGRANGIAN_H_
#pragma once


#include "ChimeraCore.h"

//Math
#include "Base/Vector2.h"
#include "Base/Vector3.h"
#include "Integration/TrajectoryIntegrators.h"
#include "Interpolation/LinearInterpolation2D.h"
#include "Integration/AdvectionIntegrator.h"

namespace Chimera {

	using namespace Core;

	namespace Advection {

		/** Solves the advection Function u* = u grad u with the Semi-Lagrangian (characteristcs) method. */
		template<template<class T> class ArrayT, class VectorType>
		class SemiLagrangianIntegrator : public AdvectionIntegrator<ArrayT, VectorType> {
			public:
				/************************************************************************/
				/* ctors                                                                */
				/************************************************************************/
				/** 
				 ** Receives the initial Array2D u and the final Array2D u*. The advection result is going to be stored
				 ** in the Array2D u*. */
				SemiLagrangianIntegrator(ArrayT<VectorType> *pVelocityField, ArrayT<VectorType> *pAuxVelocityField, const trajectoryIntegratorParams_t<VectorType> &trajectoryIntegratorParams);

				/************************************************************************/
				/* 2D Semi Lagrangian functions							                */
				/************************************************************************/
				/** Integration function updates the u velocity Array with the result of u* = u grad u. */
			    void integrateVelocityField();

				/** Integration function of the special Cells */
				void integrateVelocityFieldSpecialCells();

				/** Integrates an scalar field accordingly with the 2D velocity field.  This function already swap buffers
				 ** on the pScalarFieldBuffer. DO NOT SWAP BUFFERS EXPLICITLY AFTER IT. */
				void integrateScalarField(Core::DoubleBuffer<ArrayT<Scalar>, Scalar> *pScalarFieldBuffer);
		};
	}
}

#endif