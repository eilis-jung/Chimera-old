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

#ifndef _CHIMERA_BOUNDARY_CONDITION_FACTORY_H
#define _CHIMERA_BOUNDARY_CONDITION_FACTORY_H
#pragma  once

/************************************************************************/
/* Chimera Core                                                         */
/************************************************************************/
#include "ChimeraCore.h"

/************************************************************************/
/* Chimera Data                                                         */
/************************************************************************/
/** Boundary conditions*/
#include "Boundary Conditions/BoundaryCondition.h"
#include "Boundary Conditions/InflowBC.h"
#include "Boundary Conditions/OutflowBC.h"
#include "Boundary Conditions/NoSlipBC.h"
#include "Boundary Conditions/FreeSlipBC.h"
#include "Boundary Conditions/JetBC.h"
#include "Boundary Conditions/PeriodicBC.h"
#include "Boundary Conditions/FarFieldBC.h"

namespace Chimera {

	using namespace Core;
	
	namespace BoundaryConditions {

		/** Boundary condition factory. Loads and creates different types of boundary conditions. 
		 ** Also, this class is a Singleton. */
		
		class BoundaryConditionFactory : public Singleton<BoundaryConditionFactory> {

			/** Load one boundary condition, 2D case */
			BoundaryCondition<Vector2> * loadBoundaryCondition2D(const auto_ptr<ifstream> &fileStream, dimensions_t dimensions);

			/** Load one boundary condition, 3D case */
			BoundaryCondition<Vector3> * loadBoundaryCondition3D(const auto_ptr<ifstream> &fileStream, dimensions_t dimensions);

		public:

			/** Boundary condition loader for 2D or 3D grids.*/
			template <class VectorT>
			vector<BoundaryCondition<VectorT> *> loadBCs(const string &boundaryConditionsFile, dimensions_t dimensions) {
				Logger::get() << "Loading boundary conditions file: " << boundaryConditionsFile << endl;
				vector<BoundaryCondition<VectorT> *> boundaryConditions;
				auto_ptr<ifstream> fileStream(new ifstream(boundaryConditionsFile.c_str()));
				if(fileStream->fail())
					throw (boundaryConditionsFile);
				string fString;
				int numBoundaryConditions;
				(*fileStream) >> fString;								// Skipping numBoundaryConditions
				(*fileStream) >> numBoundaryConditions;

				while(numBoundaryConditions-- > 0) {
					//Catch errors
					Logger::get() << "Parsing boundary ID: " << boundaryConditions.size() << endl;
					if(dimensions.z == 0) {
						BoundaryCondition<Vector2> *pCondition = loadBoundaryCondition2D(fileStream, dimensions); 
						boundaryConditions.push_back((BoundaryCondition<VectorT> *) pCondition);
					} else {
						BoundaryCondition<Vector3> *pCondition = loadBoundaryCondition3D(fileStream, dimensions); 
						boundaryConditions.push_back((BoundaryCondition<VectorT> *) pCondition);
					}
				}

				Logger::get() << "Boundary conditions successfully loaded." << endl;
				return boundaryConditions;
			}


		};

	}
}

#endif