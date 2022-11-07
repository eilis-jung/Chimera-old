
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

#ifndef _RENDERING_CUT_VOXELS_WINDOW_H
#define _RENDERING_CUT_VOXELS_WINDOW_H

#pragma  once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraCutCells.h"

#include "Windows/BaseWindow.h"
#include "Visualization/VolumeMeshRenderer.h"
#include "Visualization/CutVoxelsVelocityRenderer3D.h"

namespace Chimera {
	using namespace CutCells;
	using namespace Grids;

	namespace Windows {

		template <class VectorType>
		class CutVoxelsWindow : public BaseWindow {
		public:
			#pragma region Constructors
			CutVoxelsWindow(Rendering::VolumeMeshRenderer<VectorType> *pVolumeRenderer, Rendering::CutVoxelsVelocityRenderer3D<VectorType> *pVelRenderer);
			#pragma endregion

			#pragma region AccessFunctions
			FORCE_INLINE CutVoxels3D<VectorType> * getCutVoxels() {
				return m_pCutVoxels;
			}

			FORCE_INLINE Scalar getPressure() const {
				return m_cellPressure;
			}
			FORCE_INLINE Scalar getDivergent() const {
				return m_cellDivergent;
			}
			#pragma endregion

			FORCE_INLINE void setSelectedCell(int selectedCell) {
				int prevCutCell = m_pCutVoxelsRenderer->getSelectedCutVoxel();
				m_pCutVoxelsRenderer->setSelectedCutVoxel(selectedCell);
				if(selectedCell != -1 && prevCutCell != selectedCell) {
					updateAttributes();
					//updateFaceAttributes();
				}
			}

			#pragma region UpdateFunctions
			FORCE_INLINE virtual void update() {
				
			}
			#pragma endregion

		private:

			#pragma region ClassMembers
			/** Ptr copies */
			CutVoxels3D<VectorType> *m_pCutVoxels;
			Rendering::VolumeMeshRenderer<VectorType> *m_pCutVoxelsRenderer;
			Rendering::CutVoxelsVelocityRenderer3D<VectorType> *m_pCutVelocitiesRenderer;

			/**Special cell attributes: */
			dimensions_t m_regularGridIndex;
			Scalar m_cellPressure;
			Scalar m_cellDivergent;
			bool m_drawVoxelsNeighbors;
			#pragma endregion

			#pragma region Initialization
			void initializeAntTweakBar();
			#pragma endregion

			#pragma region PrivateFunctionalities
			void updateAttributes();
			#pragma endregion
		};
	}
}

#endif