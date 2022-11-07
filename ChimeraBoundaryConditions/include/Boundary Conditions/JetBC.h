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

#pragma once

#ifndef __CHIMERA_JET_BC__
#define __CHIMERA_JET_BC__

/************************************************************************/
/* Chimera Data                                                         */
/************************************************************************/
#include "Boundary Conditions/BoundaryCondition.h"


namespace Chimera {

	namespace BoundaryConditions {

		template <class VectorT> 
		class JetBC : public BoundaryCondition<VectorT> {

			VectorT m_velocity;
			Scalar m_size;
			Scalar m_densityVariation;
			Scalar m_temperatureVariation;
			Scalar m_minDensity;
			Scalar m_minTemperature;

			/** Setups density and temperature on the grid */
			void setScalarValues(int x, int y, GridData2D *gridData);
			void setScalarValues(int x, int y, int z, GridData3D *gridData);

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
			JetBC(boundaryLocation_t boundaryLocation, 
				range1D_t boundaryRange, dimensions_t dimensions) :
			BoundaryCondition<VectorT>(Jet, boundaryLocation, boundaryRange, dimensions) { }

			inline void setParameters(const VectorT & velocity, Scalar size, Scalar temperatureVariation, Scalar densityVariation, 
										Scalar minDensity = 0.9, Scalar minTemperature = 0.9) {
				m_velocity = velocity;
				m_size = size;
				m_temperatureVariation = temperatureVariation;
				m_densityVariation = densityVariation;
				m_minTemperature = minTemperature;
				m_minDensity = minDensity;
			}
			
			Scalar getSize() const {
				return m_size;
			}
		};
	}
}

#endif