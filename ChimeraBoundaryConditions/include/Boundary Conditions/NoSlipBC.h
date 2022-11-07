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

#ifndef __CHIMERA_NO_SLIP_BC__
#define __CHIMERA_NO_SLIP_BC__
#pragma once

/************************************************************************/
/* Chimera Data                                                         */
/************************************************************************/
#include "Boundary Conditions/BoundaryCondition.h"


namespace Chimera {

	namespace BoundaryConditions {

		template <class VectorT>
		class NoSlipBC : public BoundaryCondition<VectorT> {
			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			VectorT m_solidVelocity;

			/************************************************************************/
			/*  Boundary conditions                                                 */
			/************************************************************************/
			virtual void applyBoundaryConditionNorth(GridData<VectorT> *gridData, solverType_t solverType);
			virtual void applyBoundaryConditionSouth(GridData<VectorT> *gridData, solverType_t solverType);
			virtual void applyBoundaryConditionWest(GridData<VectorT> *gridData, solverType_t solverType);
			virtual void applyBoundaryConditionEast(GridData<VectorT> *gridData, solverType_t solverType);
			virtual void applyBoundaryConditionBack(GridData<VectorT> *gridData, solverType_t solverType);
			virtual void applyBoundaryConditionFront(GridData<VectorT> *gridData, solverType_t solverType);

		public:

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			/** 2D ctor*/
			NoSlipBC(boundaryLocation_t boundaryLocation, 
				range1D_t boundaryRange, dimensions_t dimensions) :
			BoundaryCondition<VectorT>(NoSlip, boundaryLocation, boundaryRange, dimensions) { }

			/************************************************************************/
			/* Access functions                                                      */
			/************************************************************************/
			void setSolidVelocity(const VectorT &solidVelocity) {
				m_solidVelocity = solidVelocity;
			}

			const VectorT & getSolidVelocity() const {
				return m_solidVelocity;
			}
			
			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			static void enforceSolidWalls(QuadGrid *pGridData, Vector2 solidVelocity = Vector2(0, 0));
			static void enforceSolidWalls(HexaGrid *pGridData, Vector3 solidVelocity = Vector3(0, 0, 0));

		};
	}
}

#endif