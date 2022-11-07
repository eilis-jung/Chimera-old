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

#ifndef _CHIMERA_3D_PARTICLES_RENDERER_
#define _CHIMERA_3D_PARTICLES_RENDERER_
#pragma  once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraResources.h"
#include "ChimeraCutCells.h"
#include "ChimeraInterpolation.h"
#include "ChimeraParticles.h"

#include "Visualization/ScalarFieldRenderer.h"
#include "RenderingUtils.h"

namespace Chimera {
	using namespace CutCells;
	using namespace Resources;
	using namespace Interpolation;
	using namespace Particles;

	namespace Rendering {

		template <class VectorType, template <class> class ArrayType>
		class ParticlesRenderer {

		public:

			#pragma region ExternalStructures
			static const int s_maxTrailSize;

			struct renderingParams_t {
				bool draw;
				int trailSize;
				
				Scalar particleSize;
				scalarColorScheme_t colorScheme;

				Scalar minScalarfieldValue, maxScalarfieldValue;
				bool drawSelectedVoxelParticles;
				dimensions_t selectedVoxelDimension;

				bool drawParticlePerSlice;
				int gridSliceDimension;

				int selectedScalarParam;

				Scalar velocityScale;
				renderingParams_t() {
					draw = true;
					trailSize = 1;
					particleSize = 1.0f;

					colorScheme = grayscale;
					
					minScalarfieldValue = 0;
					maxScalarfieldValue = 1.0f;
					
					drawSelectedVoxelParticles = false;
					drawParticlePerSlice = false;
					gridSliceDimension = 0;
					velocityScale = 0.01f;
					selectedScalarParam = -1;
				}
			} m_renderingParams;
			#pragma endregion
			
			#pragma region Constructors
			ParticlesRenderer(ParticlesData<VectorType> *pParticlesData, renderingParams_t renderingParams, Scalar gridSpacing) {
				m_pParticlesData = pParticlesData;
				//m_pCutVoxels = nullptr;
				m_gridSpacing = gridSpacing;
				//Visualize density by default
				m_visAttribute = "density";
				m_renderingParams = renderingParams;
				m_currentTrailSize = 0;
				m_pParticlesTrails = new Array2D<VectorType>(dimensions_t(pParticlesData->getPositions().size(), s_maxTrailSize));
				m_particlesColors.resize(pParticlesData->getPositions().size() * 4, 0.0f);
				initializeInternalStructs();
				initializeVBOs();
				initializeShaders();

				//Update once to setup right density colors
				update(0);
			}
			#pragma endregion


			#pragma region CPUShadingFunctions
			Vector3 jetShading(int i, Scalar scalarFieldValue, Scalar minValue, Scalar maxValue);
			Vector3 grayScaleShading(int i, Scalar scalarFieldValue, Scalar minValue, Scalar maxValue);
			Vector3 viridisShading(int i, Scalar scalarFieldValue, Scalar minValue, Scalar maxValue);
			#pragma endregion

			#pragma region Functionalities
			void updateParticlesColorsCPU(Scalar minScalarValue, Scalar maxScalarValue);
			void updateParticlesColorsShaders(Scalar minScalarValue, Scalar maxScalarValue);
			void update(Scalar dt);
			void draw();
			#pragma endregion

			#pragma region AccessFunctions
			FORCE_INLINE renderingParams_t & getRenderingParams() {
				return m_renderingParams;
			}

			/*FORCE_INLINE void setCutVoxels(CutVoxels3D<Vector3> *pCutVoxels) {
				m_pCutVoxels = pCutVoxels;
			}*/
				
			void setVelocityInterpolant(Interpolant<VectorType, ArrayType, VectorType> *pVelocityInterpolant) {
				m_pVelocityInterpolant = pVelocityInterpolant;
			}

			Interpolant<VectorType, ArrayType, VectorType> * getVelocityInterpolant() {
				return m_pVelocityInterpolant;
			}

			string * getVisualizedAttributeStrPtr() {
				return &m_visAttribute;
			}

			ParticlesData<VectorType> * getParticlesData() {
				return m_pParticlesData;
			}

			#pragma endregion
		private:

			#pragma region ClassMembers
			/** Particles data */
			ParticlesData<VectorType> *m_pParticlesData;
			Scalar m_gridSpacing;

			/** Particles position vertex buffer object. */
			GLuint *m_pParticlesVBO;
			/** Color VBO*/
			GLuint *m_pParticlesColorVBO;
			/** Trails positions and indexes VBOs. */
			GLuint *m_pParticleTrailsVBO;
			GLuint *m_pParticleTrailsIndexesVBO;
			GLuint *m_pScalarAttributesVBO;

			/** CPU space trails */
			Array2D<VectorType> *m_pParticlesTrails;
			int m_currentTrailSize;
			int * m_pParticlesTrailsIndexes;
			/** CPU space particles color */
			vector<Scalar> m_particlesColors;

			/** Particle's attribute to visualize*/
			string m_visAttribute;

			/** Interpolant */
			Interpolant<VectorType, ArrayType, VectorType> *m_pVelocityInterpolant;

			/** Cut Cells/Voxels data*/
			/*CutCells2D<VectorType> *m_pCutCells;
			CutVoxels3D<Vector3> *m_pCutVoxels;*/

			/** Shaders data */
			//Viridis scalar color shader
			shared_ptr<GLSLShader> m_pViridisColorShader;
			GLuint m_virMinScalarLoc;
			GLuint m_virMaxScalarLoc;
			GLuint m_virAvgScalarLoc;

			//Jet scalar color shader
			shared_ptr<GLSLShader> m_pJetColorShader;
			GLuint m_jetMinScalarLoc;
			GLuint m_jetMaxScalarLoc;
			GLuint m_jetAvgScalarLoc;

			//Grayscale scalar color shader
			shared_ptr<GLSLShader> m_pGrayScaleColorShader;
			GLuint m_grayMinScalarLoc;
			GLuint m_grayMaxScalarLoc;
			#pragma endregion

			#pragma region InitializationFunctions
			void initializeInternalStructs();
			void initializeVBOs();
			void initializeShaders();
			#pragma endregion

			#pragma region PrivateUpdateFunctions
			FORCE_INLINE void applyColorShader(Scalar minValue, Scalar maxValue, Scalar avgValue) const {
				switch (m_renderingParams.colorScheme) {
				case viridis:
					m_pViridisColorShader->applyShader();
					glUniform1f(m_virMinScalarLoc, minValue);
					glUniform1f(m_virMaxScalarLoc, maxValue);
					glUniform1f(m_virAvgScalarLoc, avgValue);
					break;
				case jet:
					m_pJetColorShader->applyShader();
					glUniform1f(m_jetMinScalarLoc, minValue);
					glUniform1f(m_jetMaxScalarLoc, maxValue);
					glUniform1f(m_jetAvgScalarLoc, avgValue);
					break;

				case grayscale:
					m_pGrayScaleColorShader->applyShader();
					glUniform1f(m_grayMinScalarLoc, minValue);
					glUniform1f(m_grayMaxScalarLoc, maxValue);
					break;
				}
			}

			FORCE_INLINE void removeColorShader() const {
				switch (m_renderingParams.colorScheme) {
				case viridis:
					m_pViridisColorShader->removeShader();
					break;
				case jet:
					m_pJetColorShader->removeShader();
					break;
				case grayscale:
					m_pGrayScaleColorShader->removeShader();
					break;
				}
			}

			/** Updating */
			void updateTrails();
			void updateVBOs();
			void updateParticlesColorsGridSlice();
			#pragma endregion

			
		};
	}

	
}

#endif