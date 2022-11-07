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

#include "Windows/ParticleVisualizationWindow.h"
#include "RenderingUtils.h"
namespace Chimera {
	namespace Windows {

		template <class VectorT, template <class> class ArrayType>
		ParticleVisualizationWindow<VectorT, ArrayType>::ParticleVisualizationWindow(ParticlesRenderer<VectorT, ArrayType> *pParticlesRenderer) : BaseWindow(Vector2(16, 300), Vector2(300, 190), "Particles Visualization") {
			m_pParticlesRenderer = pParticlesRenderer;

			TwAddVarRW(m_pBaseBar, "drawParticles", TW_TYPE_BOOL8, &m_pParticlesRenderer->getRenderingParams().draw, " label='Draw particles' group='Particles'");
			TwAddVarRW(m_pBaseBar, "particleSize", TW_TYPE_FLOAT, &m_pParticlesRenderer->getRenderingParams().particleSize, " label='Particles size' group='Particles'");
			string maxTrailSizeStr = string("label='Particles Trail Size' group='Particles' min='0' max='") + intToStr(ParticlesRenderer<VectorT, ArrayType>::s_maxTrailSize - 1) + string("'");
			TwAddVarRW(m_pBaseBar, "trailSize", TW_TYPE_INT32, &m_pParticlesRenderer->getRenderingParams().trailSize, maxTrailSizeStr.c_str());

			TwEnumVal coloringTypeEV[] = { { Rendering::grayscale, "GrayScale" }, { Rendering::viridis, "Viridis" }, { Rendering::jet, "Jet" }};
			TwType coloringType = TwDefineEnum("particlesColoringType", coloringTypeEV, 3);
			TwAddVarRW(m_pBaseBar, "particlesColoringScheme", coloringType, (void *)&m_pParticlesRenderer->getRenderingParams().colorScheme, "label='Particles coloring scheme'  group='Particles'");

			vector<TwEnumVal> particlesScalarAttributes;
			ParticlesData<VectorT> *pParticlesData = pParticlesRenderer->getParticlesData();
			auto scalarMap = pParticlesData->getScalarAttributesMap();
			int index = 0;
			for (auto iter = scalarMap.begin(); iter != scalarMap.end(); iter++) {
				TwEnumVal evalTw;
				evalTw.Value = index;
				evalTw.Label = iter->first.c_str();
				particlesScalarAttributes.push_back(evalTw);
				index++;
			}
			if (index > 0) {
				m_pParticlesRenderer->getRenderingParams().selectedScalarParam = 0;
				TwType scalarAttributes = TwDefineEnum("particlesScalarAttributes", &particlesScalarAttributes[0], index);
				TwAddVarRW(m_pBaseBar, "particleSelectAttribute", scalarAttributes, (void *)&m_pParticlesRenderer->getRenderingParams().selectedScalarParam, "label='Particles selected scalar attribute'  group='Particles'");
			}
			
			TwAddVarRW(m_pBaseBar, "selectedDimParticle", TW_TYPE_BOOL8, &m_pParticlesRenderer->getRenderingParams().drawSelectedVoxelParticles, "label='Select particles by voxel'  group='Particles'");
			//TwAddVarRW(m_pBaseBar, "selectParticleGridSlice", TW_TYPE_BOOL8, &m_pParticleSystem3D->getRenderingParams().drawParticlePerSlice, "label='Select particles by grid slice'  group='Particles'");
			//TwAddVarRW(m_pBaseBar, "particleGridSlice", TW_TYPE_INT32, &m_pParticleSystem3D->getRenderingParams().gridSliceDimension, "Ith grid slice'  group='Particles'");
		}
		
		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		template <class VectorT, template <class> class ArrayType>
		void ParticleVisualizationWindow<VectorT, ArrayType>::update() {
			
		}

		/************************************************************************/
		/* GridVisualization declarations - Linking time                        */
		/************************************************************************/
		template ParticleVisualizationWindow<Vector2, Array2D>;
		template ParticleVisualizationWindow<Vector3, Array3D>;

	}
}