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

#ifndef __CHIMERA_BOUNDARY_CONDITION__
#define __CHIMERA_BOUNDARY_CONDITION__

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraPoisson.h"

using namespace std;

namespace Chimera {

	using namespace Core;
	using namespace Grids;
	using namespace Poisson;

	namespace BoundaryConditions {

		/** Boundary conditions types. */
		typedef enum boundaryType_t {
			Outflow,
			Inflow,
			NoSlip,
			FreeSlip,
			Jet,
			Periodic,
			Farfield
		} boundaryType_t;

		/* Encapsulates the base class for boundary conditions. The class VectoT passed as a template dictates wether 
		 * the grid is 2D or 3D.*/

		template <class VectorT>
		class BoundaryCondition {

		protected:

			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			boundaryType_t m_boundaryType;
			boundaryLocation_t m_boundaryLocation;
			range1D_t m_boundaryRange;
			dimensions_t m_dimensions;

			/************************************************************************/
			/* Poisson Matrix update                                                */
			/************************************************************************/
			virtual void updateWestBoundary(PoissonMatrix *pPoissonMatrix);
			virtual void updateEastBoundary(PoissonMatrix *pPoissonMatrix);
			virtual void updateNorthBoundary(PoissonMatrix *pPoissonMatrix);
			virtual void updateSouthBoundary(PoissonMatrix *pPoissonMatrix);
			virtual void updateFrontBoundary(PoissonMatrix *pPoissonMatrix);
			virtual void updateBackBoundary(PoissonMatrix *pPoissonMatrix);
		

			/************************************************************************/
			/*  Boundary conditions                                                 */
			/************************************************************************/
			virtual void applyBoundaryConditionNorth(GridData<VectorT> *gridData, solverType_t solverType) = 0;
			virtual void applyBoundaryConditionSouth(GridData<VectorT> *gridData, solverType_t solverType) = 0;
			virtual void applyBoundaryConditionWest(GridData<VectorT> *gridData, solverType_t solverType) = 0;
			virtual void applyBoundaryConditionEast(GridData<VectorT> *gridData, solverType_t solverType) = 0;
			virtual void applyBoundaryConditionBack(GridData<VectorT> *gridData, solverType_t solverType) = 0;
			virtual void applyBoundaryConditionFront(GridData<VectorT> *gridData, solverType_t solverType) = 0;	


		public:

			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			/** Default ctor */
			BoundaryCondition(boundaryType_t boundaryType, boundaryLocation_t boundaryLocation, range1D_t boundaryRange, dimensions_t dimensions)
				: m_dimensions(dimensions), m_boundaryType(boundaryType), m_boundaryLocation(boundaryLocation), m_boundaryRange(boundaryRange) {
			}

			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			boundaryType_t getType() const {
				return m_boundaryType;
			}

			boundaryLocation_t getLocation() const {
				return m_boundaryLocation;
			}
			range1D_t getRange() const {
				return m_boundaryRange;
			}

			const dimensions_t & getDimensions() const {
				return m_dimensions;
			}

			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
			virtual void zeroVelocity(GridData<VectorT> *gridData);
			virtual void applyBoundaryCondition(GridData<VectorT> *gridData, solverType_t solverType);
			static void zeroSolidBoundaries(GridData<VectorT> * pGridData);
			static void updateSolidWalls(PoissonMatrix *pPoissonMatrix, Array<char> solidWalls = Array<char>(dimensions_t(0, 0, 0)), bool padPoissonMatrix = true);

			/************************************************************************/
			/* Poisson Matrix update                                                */
			/************************************************************************/
			/** Updates poissonMatrix according with the boundary condition */
			virtual void updatePoissonMatrix(PoissonMatrix *pPoissonMatrix);


			static void fixAllNeumannConditions(PoissonMatrix *pPoissonMatrix, vector<BoundaryCondition<VectorT>*> boundaryConditions);

			
		};
	}
}



#endif
