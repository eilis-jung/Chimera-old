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

#ifndef _RENDERING_GRID_VISUALIZATION_WINDOW_H
#define _RENDERING_GRID_VISUALIZATION_WINDOW_H

#pragma  once

#include "ChimeraCore.h"
#include "ChimeraResources.h"
#include "Windows/BaseWindow.h"

//Rendering Cross-ref
#include "Visualization/QuadGridRenderer.h"
#include "Particles/ParticleSystem2D.h"
#include "Particles/ParticleSystem3D.h"

namespace Chimera {
	using namespace Resources;
	using namespace Rendering;
	namespace Windows {

		template <class VectorT>
		class GridVisualizationWindow : public BaseWindow {


		private:
			#pragma region ClassMembers
			GridRenderer<VectorT> *m_pGridRenderer;

			/** Grid */
			bool m_drawSolidCells;
			bool m_drawRegularCells;

			/** Grid Vis: 3-D only */
			/** Draw grid countours*/
			bool m_drawGridPlaneYZ;
			bool m_drawGridPlaneXZ;
			bool m_drawGridPlaneXY;

			int m_numOfGridPlanesDrawn;

			/** Visualize scalar fields */
			bool m_drawScalarFieldSlices;
			bool m_drawVelocitySlices;
			dimensions_t m_ithRenderingPlanes;

			/** Grid statistics */
			dimensions_t m_gridDimensions;
			int m_totalGridCells;

			/** Particle system */
			ParticleSystem2D *m_pParticleSystem;
			
			/** Visualization type*/
			scalarVisualization_t m_scalarDrawingType;
			vectorVisualization_t m_vectorFieldDrawingType;
			#pragma endregion

		public:
			/************************************************************************/
			/* ctors                                                                */
			/************************************************************************/
			GridVisualizationWindow(GridRenderer<VectorT> *pGridRenderer);

			#pragma region AccessFunctions
			FORCE_INLINE GridRenderer<VectorT> * getGridRenderer() const {
				return m_pGridRenderer;
			}

			/** Grid visualization */
			FORCE_INLINE bool drawGridSolidCells() const {
				return m_drawSolidCells;
			}

			FORCE_INLINE bool drawRegularGridCells() const {
				return m_drawRegularCells;
			}

			FORCE_INLINE scalarVisualization_t getScalarFieldType() const {
				return m_scalarDrawingType;
			}

			FORCE_INLINE vectorVisualization_t getVelocityDrawingType() const {
				return m_vectorFieldDrawingType;
			}

			const dimensions_t & getIthRenderingPlanes() const {
				return m_ithRenderingPlanes;
			}

			FORCE_INLINE bool drawYZPlaneSlice() const {
				return m_drawGridPlaneYZ;
			}
			FORCE_INLINE bool drawXZPlaneSlice() const {
				return m_drawGridPlaneXZ;
			}
			FORCE_INLINE bool drawXYPlaneSlice() const {
				return m_drawGridPlaneXY;
			}

			FORCE_INLINE bool drawScalarFieldSlices() const {
				return m_drawScalarFieldSlices;
			}

			FORCE_INLINE bool drawVelocitySlices() const {
				return m_drawVelocitySlices;
			}

			FORCE_INLINE int getNumberOfDrawnPlanes() const {
				return m_numOfGridPlanesDrawn;
			}
			#pragma endregion

			#pragma region Functionalities
			virtual void update();

			void addParticleSystem(ParticleSystem2D *pParticleSystem);
			void addParticleSystem(ParticleSystem3D *pParticleSystem);
			#pragma endregion
		};
	}
}

#endif