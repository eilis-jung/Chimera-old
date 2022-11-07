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


#ifndef _CHIMERA_ISOCOUNTOUR_RENDERER_H_
#define _CHIMERA_ISOCOUNTOUR_RENDERER_H_
#pragma once


#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraLevelSets.h"
#include "Windows/BaseWindow.h"



namespace Chimera {

	using namespace Grids;
	using namespace Windows;

	namespace Rendering {

		class IsocontourRenderer {
		public:
			
			#pragma region ClassDefinitions 
			typedef struct params_t {
				bool drawIsocontours;
				bool drawIsopoints;

				int numIsocontours;
				Scalar initialIsoValue;
				Scalar isoStepVertical;
				Scalar isoStepHorizontal;

				BaseWindow::scalarVisualization_t currentScalarVisualization;
				
				params_t() {
					drawIsocontours = true;
					drawIsopoints = false;
					numIsocontours = 1;
					initialIsoValue = 0.001;
					isoStepVertical = 0.001;
					isoStepHorizontal = 0.01;
					currentScalarVisualization = BaseWindow::drawNoScalarField;
				}
			};
			
			#pragma endregion ClassDefinitions

			#pragma region ConstructorsDestructors
			IsocontourRenderer(GridData2D *pGridData);
			#pragma endregion ConstructorsDestructors 

			#pragma region AccessFunctions
			vector<vector<Vector2>> & getIsolines() {
				return m_isoLines;	
			}
			params_t & getParams() {
				return m_params;
			}
			#pragma endregion AccessFunctions

			#pragma region DrawingFunctions
			void drawIsocontours() const;
			void drawIsoPoints() const;
			#pragma endregion DrawingFunctions

			#pragma region UpdateFunctions
			void update();
			#pragma endregion 
		private:
			

			#pragma region ClassMembers 
			GLuint *m_pPointsVBO;
			vector<vector<Vector2>> m_isoLines;
			params_t m_params;
			params_t m_oldParams;
			GridData2D *m_pGridData;
			#pragma endregion ClassMembers

			#pragma region VBOsFunctions
			void initializeVBOs();
			void updateVBOs();
			#pragma endregion VBOsFunctions
			
			#pragma region AuxiliaryFunctions 
			void generateIsolines();
			#pragma endregion AuxiliaryFunctions

		};
	}
}

#endif
