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

#ifndef __CHIMERA_CUT_VOXELS_SOLVER_3D__
#define __CHIMERA_CUT_VOXELS_SOLVER_3D__
#pragma once

#include "ChimeraCore.h"
#include "ChimeraAdvection.h"
#include "ChimeraParticles.h"
#include "ChimeraInterpolation.h"
#include "ChimeraLevelSets.h"

#include "Solvers/FlowSolver.h"

namespace Chimera {

	using namespace Core;
	using namespace Advection;
	using namespace Solvers;
	using namespace Particles;
	using namespace LevelSets;

	/** Implementation of the classic Navier-Stokes solver, for unsteady incompressible flows.
	It uses the following configurations:
	Dependent variables: Pressure, Cartesian velocity components;
	Variable arrangement: Staggered;
	Pressure Coupling: Fractional Step;
	*/

	namespace Solvers {
		class CutVoxelSolver3D : public FlowSolver<Vector3, Array3D> {

		
			public:
		
				#pragma region Constructors
				//Default constructor for derived classes
				CutVoxelSolver3D(const params_t &params, StructuredGrid<Vector3> *pGrid,
									const vector<BoundaryCondition<Vector3> *> &boundaryConditions = vector<BoundaryCondition<Vector3> *>(),
									const vector<PolygonalMesh<Vector3> *> &polygonMeshes = vector<PolygonalMesh<Vector3> *>());
				#pragma endregion 

				#pragma region UpdateFunctions
				/**Overrides FlowSolver's update function in order to perform advection & cloth movement correctly */
				void update(Scalar dt) override;

				/** Updates thin objects Poisson Matrix */
				void updatePoissonThinSolidWalls();

				/** Updates interal cut-cells divergence */
				void updateCutCellsDivergence(Scalar dt);

				void applyForces(Scalar dt) override { };
				#pragma endregion

				#pragma region AccessFunctions
				CutVoxels3D<Vector3> * getCutVoxels() {
					return m_pCutVoxels;
				}
				#pragma endregion 

				#pragma region ObjectsInitialization
				/** Reinitializes cut-cell structures after the update of any object present on the simulation domain */
				void reinitializeThinBounds() { };
				#pragma endregion 
	
			protected:
				typedef Interpolant <Vector3, Array3D, Vector3> VelocityInterpolant;
				typedef Interpolant <Scalar, Array3D, Vector3> ScalarInterpolant;

				#pragma region ClassMembers
				/** Polygonal meshes */
				vector<PolygonalMesh<Vector3> *> m_polyMeshesVec;

				/** GridData shortcut*/
				GridData3D *m_pGridData;

				/** CutVoxels */
				CutVoxels3D<Vector3> *m_pCutVoxels;

				/*Cut-voxels pressures and divergents */
				vector<Scalar> m_cutCellsPressures;
				vector<Scalar> m_cutCellsDivergents;

				/** Nodal based velocities */
				Array3D<Vector3> m_nodalBasedVelocities;
				Array3D<Vector3> m_auxNodalBasedVelocities;

				/** Particle-based advection */
				GridToParticles<Vector3, Array3D> *m_pGridToParticlesTransfer;
				ParticlesToGrid<Vector3, Array3D> *m_pParticlesToGridTransfer;

				/** Velocity data for cut-voxes */
				CutVoxelsVelocities3D *m_pCutVoxelsVelocities3D;
				/** Storing intermediary velocity data for cut-voxes */
				CutVoxelsVelocities3D *m_pAuxCutVoxelsVelocities3D;
				#pragma endregion


				#pragma region InitializationFunctions
				/**Initializes cut-cell structure */
				CutVoxels3D<Vector3>* initializeCutVoxels();

				/** Initializes the Poisson matrix accordingly with Finite Differences formulation */
				PoissonMatrix * createPoissonMatrix();

				/** Initializes several interpolants used for the simulation and debugging */
				void initializeInterpolants();

				/** Create grid to particles and particles to grid class transfers */
				GridToParticles<Vector3, Array3D> * createGridToParticles();
				ParticlesToGrid<Vector3, Array3D> * createParticlesToGrid();
				#pragma endregion

				#pragma region InternalAuxFunctions
				int getRowIndex(const dimensions_t &currDim, halfFaceLocation_t faceLocation) {
					if (currDim.x - 2 < 0 || currDim.y - 2 < 0 || currDim.z - 2 < 0)
						throw(exception("CutVoxelSolver3D getRowIndex: invalid currDim, too close to boundaries"));
					switch (faceLocation) {
					case rightHalfFace:
						return m_pPoissonMatrix->getRowIndex(currDim.x, currDim.y - 1, currDim.z - 1);
						break;
					case bottomHalfFace:
						return m_pPoissonMatrix->getRowIndex(currDim.x - 1, currDim.y - 2, currDim.z - 1);
						break;
					case backHalfFace:
						return m_pPoissonMatrix->getRowIndex(currDim.x - 1, currDim.y - 1, currDim.z - 2);
						break;
					case leftHalfFace:
						return m_pPoissonMatrix->getRowIndex(currDim.x - 2, currDim.y - 1, currDim.z - 1);
						break;
					case topHalfFace:
						return m_pPoissonMatrix->getRowIndex(currDim.x - 1, currDim.y, currDim.z - 1);
						break;
					case frontHalfFace:
						return m_pPoissonMatrix->getRowIndex(currDim.x - 1, currDim.y - 1, currDim.z);
						break;
					default:
						return -1;
						break;
					}
				}
				#pragma endregion


				#pragma region PressureProjection
				/** Calculates the intermediary divergent (before projection) */
				Scalar calculateFluxDivergent(int i, int j, int k) override;

				/** Given the pressure solved by the Linear system, projects the velocity in its divergence-free part. */
				void divergenceFree(Scalar dt) override;

				Scalar projectCutCellVelocity(Face<Vector3> *pFace, const dimensions_t &voxelLocation, velocityComponent_t velocityComponent, Scalar dt);
				#pragma endregion

				#pragma region BoundaryConditions
				/** Enforces solid walls boundary conditions */
				void enforceSolidWallsConditions(const Vector3 &solidVelocity) { };
		
				/** Enforces configuration-based scalar fields at each time-step*/
				void enforceScalarFieldMarkers() { };
				#pragma endregion

				#pragma region Advection
				void flipAdvection(Scalar dt);
				#pragma endregion

				#pragma region InternalUpdateFunctions
				void updateVorticity() { };
				#pragma endregion

			};
		
	}
	
}

#endif