//  Copyright (c) 2017, Vinicius Costa Azevedo
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

#ifndef __CHIMERA_LIQUID_REPRESENTATION_2D_H_
#define __CHIMERA_LIQUID_REPRESENTATION_2D_H_

#pragma once

#include "ChimeraCore.h"
#include "ChimeraParticles.h"
#include "LevelSets/MarchingSquares.h"

namespace Chimera {

	using namespace Particles;
	namespace LevelSets {
		
		/** This class abstracts liquids representation in 2-D. It handles the underlying FLIP particles position and
		**	possible reseeding, high resolution grid that is used to discretize the level-set function and multiple
			connected lines that represent the liquids surface */

		template <class VectorT>
		class LiquidRepresentation2D {

		
		public:
			#pragma region InternalStructures
			/** Standard parameters structure for configuring class initial parameters */
			typedef struct params_t {
				/** Number of fluid solver grid subdivisions that will generate the high-resolution level set grid */
				unsigned int levelSetGridSubdivisions;
			
				/** Pointer to particlesData which will represent liquids surface */
				ParticlesData<VectorT> *pParticlesData;

				/** Pointer to gridData structure used in the fluid solver */
				GridData2D *pGridData;

				/** Initial line meshes used to tag particles inside the fluid */
				vector<LineMesh<VectorT> *> initialLineMeshes;
			} params_t;

			typedef enum levelSetCellType_t {
				fluidCell, //Filled by particles
				boundaryCell, //Boundary between air and fluid cells
				airCell //No particles
			} levelSetCellType_t;
			#pragma endregion


			#pragma region Constructors
			LiquidRepresentation2D(const params_t &params) : m_params(params),
				m_levelSetGrid(dimensions_t(params.pGridData->getDimensions().x*pow(2, params.levelSetGridSubdivisions),
											params.pGridData->getDimensions().y*pow(2, params.levelSetGridSubdivisions))),
				m_levelSetGridWeights(dimensions_t(params.pGridData->getDimensions().x*pow(2, params.levelSetGridSubdivisions),
											params.pGridData->getDimensions().y*pow(2, params.levelSetGridSubdivisions))),
				m_levelSetCellTypes(dimensions_t(params.pGridData->getDimensions().x*pow(2, params.levelSetGridSubdivisions),
											params.pGridData->getDimensions().y*pow(2, params.levelSetGridSubdivisions))),
				m_averageParticlePositions(dimensions_t(params.pGridData->getDimensions().x*pow(2, params.levelSetGridSubdivisions),
											params.pGridData->getDimensions().y*pow(2, params.levelSetGridSubdivisions))) {
			
				Scalar dx = m_params.pGridData->getScaleFactor(0, 0).x;
				m_levelSetGridSpacing = dx / pow(2, m_params.levelSetGridSubdivisions);
				
				updateParticleTags();
				updateMeshes();
			}
			#pragma endregion
	
			#pragma region UpdateFunctions

			/*Update meshes representation to FLIP particles position update*/
			void updateMeshes();
			/** Updates cell types */
			void updateCellTypes();

			#pragma endregion

			#pragma region AccessFunctions
			Array2D<Scalar> * getLevelSetGridPtr() {
				return &m_levelSetGrid;
			}
			const Array2D<Scalar> & getLevelSetArray() {
				return m_levelSetGrid;
			}

			Scalar getLevelSetGridSpacing() const {
				return m_levelSetGridSpacing;
			}

			const vector<LineMesh<Vector2> *> & getLineMeshes() {
				return m_lineMeshes;
			}

			const Array2D<levelSetCellType_t> & getGridCellTypes() const {
				return m_levelSetCellTypes;
			}

			const params_t & getParams() const {
				return m_params;
			}

			params_t & getParams() {
				return m_params;
			}
			#pragma endregion
		private:
			#pragma region ClassMembers
			/** Parameters structure */
			params_t m_params;

			/** High-resolution scalar-field that is used to represent the level-set function */
			Array2D<Scalar> m_levelSetGrid;

			/** High-resolution scalar-field weights used to construct level-set function */
			Array2D<Scalar> m_levelSetGridWeights;

			/** Average particle positions around level set grid nodes */
			Array2D<Vector2> m_averageParticlePositions;

			/** Marker array for level set cell types */
			Array2D<levelSetCellType_t> m_levelSetCellTypes;

			/** Level set grid spacing */
			Scalar m_levelSetGridSpacing;

			/** Closed-line meshes that enclose distinct regions of the flow */
			vector<LineMesh<Vector2> *> m_lineMeshes;

			/** Map of boundary cells */
			map<int, dimensions_t> m_boundaryCellsMap;

			vector<dimensions_t> m_boundaryCells;
			#pragma endregion

			#pragma region ParticlesFunctions
			/** Splats the particle "densities" into the level-set grid using a radially invariant function */
			void particlesToLevelSetGrid();

			/** Liquid radial kernel used to splat particles values */
			FORCE_INLINE Scalar calculateLiquidKernel(Scalar distance) {
				Scalar s = (1 - distance*distance);
				return std::max(0.f, pow(s, 3));
			}

			void updateParticleTags();
			#pragma endregion
		
			#pragma region PrivateFunctionalities
			void removeBoundaryMapEntry(const dimensions_t &boundaryIndex);

			void resetLevelSetArrays();
			#pragma endregion
		};
	}
}

#endif
