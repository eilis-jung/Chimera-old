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

#ifndef __CHIMERA_CUT_CELLS_VELOCITIES_H_
#define __CHIMERA_CUT_CELLS_VELOCITIES_H_
#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"

#include "CutCells/CutCellsBase.h"
#include "ChimeraMesh.h"


namespace Chimera {
	using namespace Core;
	using namespace Grids;
	using namespace Meshes;

	namespace CutCells {

		template <class VectorT, template <class> class ElementType>
		class CutCellsVelocities {
		public:

			#pragma region Constructors
			CutCellsVelocities(Mesh<VectorT, ElementType> *pMesh, solidBoundaryType_t solidBoundary)  {
				m_pMesh = pMesh;
				m_solidBoundary = solidBoundary;
			}
			#pragma endregion
			
			#pragma region AccessFunctions

			Mesh<VectorT, ElementType> * getMesh() {
				return m_pMesh;
			}

			virtual void zeroVelocities() {
				VectorT zeroVelocity;
				if (m_pMesh) {
					for (int i = 0; i < m_pMesh->getVertices().size(); i++) {
						m_pMesh->getVertices()[i]->setVelocity(zeroVelocity);
					}
				}
			}

			virtual void zeroWeights() {
				if (m_pMesh) {
					for (int i = 0; i < m_pMesh->getVertices().size(); i++) {
						m_pMesh->getVertices()[i]->setWeight(0.0);
					}
				}
			}

			const solidBoundaryType_t & getSolidBoundaryType() const {
				return m_solidBoundary;
			}

			#pragma endregion
		protected:

			/** From the base mesh we are able to access all the vertices, and perform the necessary operations that this
				helper class supports. */
			Mesh<VectorT, ElementType> *m_pMesh;

			/** Solid boundary. Useful for processing free-slip/no-slip velocities */
			solidBoundaryType_t m_solidBoundary;


		};
	}
}


#endif
