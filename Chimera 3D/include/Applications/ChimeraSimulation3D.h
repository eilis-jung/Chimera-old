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

#ifndef _CHIMERA_CHIMERA_SIMULATION_3D_APP_H_
#define _CHIMERA_CHIMERA_SIMULATION_3D_APP_H_
#pragma  once

/************************************************************************/
/* Data                                                           */
/************************************************************************/
#include "Applications/Application3D.h"
#include "Primitives/Circle.h"

/************************************************************************/
/* Chimera 3D                                                           */
/************************************************************************/
#include "Physics/RegularGridSolver3D.h"
#include "Physics/CurvilinearGridSolver3D.h"

namespace Chimera {

	class ChimeraSimulation3D : public Application3D {

	public:
		/************************************************************************/
		/* Internal structs                                                     */
		/************************************************************************/
		typedef enum pressureInterpolationMode_t {
			interpolateDirchlet,
			interpolateNeumann
		} pressureInterpolationMode_t;


		typedef enum pressureInterpolationType_t {
			pressureBilinearInterpolation,
			pressureCubicInterpolation
		};

		/************************************************************************/
		/* ctors                                                                */
		/************************************************************************/
		ChimeraSimulation3D(int argc, char** argv, TiXmlElement *pChimeraConfig);

		/************************************************************************/
		/* Functionalities                                                      */
		/************************************************************************/
		void draw();
		void update();
	
		/************************************************************************/
		/* Call-backs                                                           */
		/************************************************************************/
		void keyboardCallback(unsigned char key, int x, int y);
		void keyboardUpCallback(unsigned char key, int x, int y);
		void specialKeyboardCallback(int key, int x, int y);
		void specialKeyboardUpCallback(int key, int x, int y);

	private:
			
			
		/** Multigrid simulation can have various configurations, which are stored
		/** in a config map. */
		map<string, SimulationConfig<Vector3> *> m_simConfigsMap;

		int m_boundaryThreshold;

		//Rotational and translational constants
		Scalar m_maxAngularSpeed;
		Scalar m_maxTranslationalSpeed;
		Scalar m_maxTranslationalAcceleration;

		//Particle System
		int m_innerIterations;
		int m_outerIterations;
		int m_boundarySmoothingIterations;
		int m_boundarySmoothingLayers;

		vector<dimensions_t> m_boundaryIndices;

		map<int, Scalar> m_minDistanceMap;
		map<int, Scalar> m_minDistanceVelocityMap;

		pressureInterpolationType_t m_pressureInterpolationType;

		Vector3 maxNonRegularBoundary;
		bool m_dumpedData;
	
		/************************************************************************/
		/* Loading functions                                                    */
		/************************************************************************/
		void loadObjects(TiXmlElement *pObjectsNode);
		void loadBackgroundGrid();
		void loadGrids();
		void loadPhysics();
		void loadParticleSystem(TiXmlElement *pParticleSystemNode);
		void loadDensityField(TiXmlElement *pDensityFieldNode);
		void loadSolidCells(TiXmlElement *pObjectsNode);
		void setupCamera(TiXmlElement *pCameraNode);

		/************************************************************************/
		/* Grid holes and boundaries                                            */
		/************************************************************************/
		// Updates the application state according grid's relative movement. Calls sub-functions updateGridMarkers and
		// updateGridBoundaries.
		void updateGridHoles(HexaGrid *pBackgrid, HexaGrid *pFrontgrid, int numBoundaryCells);
		
		// Updates cell markers mappers on both grids
		// On 3D version, I'm only using pressure boundary markers.
		void updateGridMarkers(HexaGrid *pBackgrid, HexaGrid *pFrontgrid, int numBoundaryCells);
		
		// Updates the cell map which stores the cells that are going to be extra-smoothed on multigrid phase
		void updateGridBoundaries(HexaGrid *pBackgrid, int numBoundaryCells);
		

		/************************************************************************/
		/* Private functionalities                                              */
		/************************************************************************/
		int getBottomBoundaryLimit(HexaGrid *pHexaGrid);
		int getTopBoundaryLimit(HexaGrid *pHexaGrid);
		void smoothBoundaries(int numIterations, SimulationConfig<Vector3> *pSimCfg);
		void smoothNonRegularBoundaries(int numIterations, SimulationConfig<Vector3> *pSimCfg);

		vector<Data::BoundaryCondition<Vector3> *> chimeraBoundaryConditions(StructuredGrid<Vector3> *pGrid);

		void backToFrontVelocityInterpolation(HexaGrid *pBackGrid, HexaGrid *pFrontGrid, int numBoundaryCells);
		void frontToBackVelocityInterpolation(HexaGrid *pBackGrid, HexaGrid *pFrontGrid, int numBoundaryCells);

		void backToFrontPressureInterpolation(SimulationConfig<Vector3> * pBackSimCfg, SimulationConfig<Vector3> * pFrontSimCfg);
		void frontToBackPressureInterpolation(SimulationConfig<Vector3> * pBackSimCfg, SimulationConfig<Vector3> * pFrontSimCfg);

		void frontToBackDensityInterpolation(SimulationConfig<Vector3> * pBackSimCfg, SimulationConfig<Vector3> * pFrontSimCfg);
		void backToFrontDensityInterpolation(SimulationConfig<Vector3> * pBackSimCfg, SimulationConfig<Vector3> * pFrontSimCfg);
	};
}
#endif