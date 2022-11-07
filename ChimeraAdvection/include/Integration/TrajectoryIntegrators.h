////  Copyright (c) 2013, Vinicius Costa Azevedo
////	All rights reserved.
////
////	Redistribution and use in source and binary forms, with or without
////	modification, are permitted provided that the following conditions are met: 
////
////1. Redistributions of source code must retain the above copyright notice, this
////	list of conditions and the following disclaimer. 
////	2. Redistributions in binary form must reproduce the above copyright notice,
////	this list of conditions and the following disclaimer in the documentation
////	and/or other materials provided with the distribution. 
////
////	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
////	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
////	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
////	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
////	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
////	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
////LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
////	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
////	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
////	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////
////	The views and conclusions contained in the software and documentation are those
////	of the authors and should not be interpreted as representing official policies, 
////	either expressed or implied, of the FreeBSD Project.
//
//#ifndef _MATH_TRAJECTORY_INTEGRATORS
//#define _MATH_TRAJECTORY_INTEGRATORS
//#pragma once
//
//#include "ChimeraCore.h"
//
///************************************************************************/
///* Chimera Math                                                         */
///************************************************************************/
//#include "Utils/MathUtilsCore.h"
//#include "Matrix/Matrix2x2.h"
//#include "Matrix/Matrix3x3.h"
//
//namespace Chimera {
//	
//	namespace Data {
//		class TriangleMesh;
//		class PolygonMesh3D;
//		template<class VectorT>
//		class Mesh3D;
//		class CutCells2D;
//		class CutCells3D;
//		class GridData3D;
//	}
//
//	namespace Advection {
//
//		typedef enum integrationMethod_t {
//			RungeKutta_2,
//			RungeKutta_4,
//			RungeKutta_Adaptive
//		} integrationMethod_t;
//		
//		template <typename VectorType>
//		struct trajectoryIntegratorParams_t {
//			/** Particles' initial velocity and position */
//			VectorType initialVelocity;
//			VectorType initialPosition;
//			
//			/**2D and 3D velocities field */
//			const Core::Array2D<Vector2> *pVelocityField2D;
//			const Core::Array3D<Vector3> *pVelocityField3D;
//
//			/**2D and 3D aux field */
//			const Core::Array2D<Vector2> *pAuxVelocityField2D;
//			const Core::Array3D<Vector3> *pAuxVelocityField3D;
//
//			/**2D and 3D transformation matrices*/
//			const Core::Array2D<Matrix2x2> *pTransformationMatrices2D;
//			const Core::Array3D<Matrix3x3> *pTransformationMatrices3D;
//
//			/**2D and 3D scale factors*/
//			const Core::Array2D<Vector2> *pScaleFactors2D;
//			const Core::Array3D<Vector3> *pScaleFactors3D;
//		
//			/** Node velocity field 2-D */
//			nodeVelocityField2D_t *pNodeVelocityField2D;
//			/** Node velocity field 3-D*/
//			nodeVelocityField3D_t *pNodeVelocityField3D;
//
//			/** CutCells */
//			Data::CutCells2D *pCutCells2D;
//			Data::CutCells3D *pCutCells3D;
//
//			/** Integration time-step */
//			Scalar dt;
//			
//			/** If the domain is curvilinear, we must transform the velocity accordingly*/
//			bool transformVelocity;
//			/** Used for periodic domains */
//			bool periodicDomain;
//			/** Used for backwards integration step*/
//			bool backwardsIntegration;
//
//			/** Integration methods: rungeKutta2, adaptiveRungeKutta, heun */
//			integrationMethod_t integrationMethod;
//
//			/************************************************************************/
//			/* ctors                                                                */
//			/************************************************************************/
//			trajectoryIntegratorParams_t(const Core::Array2D<Vector2> *g_pVelocityField2D, const Core::Array2D<Vector2> *g_pAuxVelocityField2D, 
//											const Core::Array2D<Matrix2x2> *g_pTransformationMatrices2D, const Core::Array2D<Vector2> *g_pScaleFactors2D) {
//												
//				pVelocityField2D = g_pVelocityField2D;
//				pAuxVelocityField2D = g_pAuxVelocityField2D;
//				pTransformationMatrices2D = g_pTransformationMatrices2D;
//				pScaleFactors2D = g_pScaleFactors2D;
//				pNodeVelocityField2D = NULL;
//				pNodeVelocityField3D = NULL;
//
//				dt = 0;
//				integrationMethod = RungeKutta_2;
//
//				transformVelocity = false;
//				periodicDomain = false;
//				backwardsIntegration = false;
//										
//			}
//
//			trajectoryIntegratorParams_t(const Core::Array3D<Vector3> *g_pVelocityField3D, const Core::Array3D<Vector3> *g_pAuxVelocityField3D, 
//											const Core::Array3D<Matrix3x3> *g_pTransformationMatrices3D, const Core::Array3D<Vector3> *g_pScaleFactors3D) {
//				pVelocityField3D = g_pVelocityField3D;
//				pAuxVelocityField3D = g_pAuxVelocityField3D;
//				pTransformationMatrices3D = g_pTransformationMatrices3D;
//				pScaleFactors3D = g_pScaleFactors3D;
//				pNodeVelocityField3D = NULL;
//
//				dt = 0;
//				integrationMethod = RungeKutta_2;
//				pNodeVelocityField2D = NULL;
//
//				transformVelocity = false;
//				periodicDomain = false;
//				backwardsIntegration = false;
//			}
//
//			trajectoryIntegratorParams_t(Data::CutCells2D *gCutCells2D, nodeVelocityField2D_t *g_pNodeVelocityField2D) {
//					pCutCells2D = gCutCells2D;
//					pCutCells3D = NULL;
//					pVelocityField2D = NULL;
//					pAuxVelocityField2D = NULL;
//					pTransformationMatrices2D = NULL;
//					pScaleFactors2D = NULL;
//					pVelocityField3D = NULL;
//					pAuxVelocityField3D = NULL;
//					pTransformationMatrices3D = NULL;
//					pScaleFactors3D = NULL;
//					pNodeVelocityField2D = g_pNodeVelocityField2D;
//
//					dt = 0;
//					integrationMethod = RungeKutta_2;
//					pNodeVelocityField3D = NULL;
//
//					transformVelocity = false;
//					periodicDomain = false;
//					backwardsIntegration = false;
//			}
//
//			trajectoryIntegratorParams_t(Data::CutCells3D *gCutCells3D, nodeVelocityField3D_t *g_pNodeVelocityField3D) {
//				pCutCells3D = gCutCells3D;
//				pCutCells2D = NULL;
//				pVelocityField2D = NULL;
//				pAuxVelocityField2D = NULL;
//				pTransformationMatrices2D = NULL;
//				pScaleFactors2D = NULL;
//				pVelocityField3D = NULL;
//				pAuxVelocityField3D = NULL;
//				pTransformationMatrices3D = NULL;
//				pScaleFactors3D = NULL;
//
//				pNodeVelocityField3D = g_pNodeVelocityField3D;
//
//				dt = 0;
//				integrationMethod = RungeKutta_2;
//				pNodeVelocityField2D = NULL;
//
//				transformVelocity = false;
//				periodicDomain = false;
//				backwardsIntegration = false;
//			}
//
//			/** Copy ctor */
//			trajectoryIntegratorParams_t(const trajectoryIntegratorParams_t &rhs) {
//				initialVelocity = rhs.initialVelocity;
//				initialPosition = rhs.initialPosition;
//
//				pVelocityField2D = rhs.pVelocityField2D;
//				pVelocityField3D = rhs.pVelocityField3D;
//
//				pTransformationMatrices2D = rhs.pTransformationMatrices2D;
//				pTransformationMatrices3D = rhs.pTransformationMatrices3D;
//
//				pScaleFactors2D = rhs.pScaleFactors2D;
//				pScaleFactors3D = rhs.pScaleFactors3D;
//
//				pNodeVelocityField2D = rhs.pNodeVelocityField2D;
//				pNodeVelocityField3D = rhs.pNodeVelocityField3D;
//
//				pCutCells2D = rhs.pCutCells2D;
//				pCutCells3D = rhs.pCutCells3D;
//
//				dt = rhs.dt;
//
//				transformVelocity = rhs.transformVelocity;
//
//				periodicDomain = rhs.periodicDomain;
//
//				integrationMethod = rhs.integrationMethod;
//				backwardsIntegration = rhs.backwardsIntegration;
//			}
//		};
//		/************************************************************************/
//		/* Local utils                                                          */
//		/************************************************************************/
//		FORCE_INLINE void limitToGridRange(Vector2 &pointPosition, int &i, int &j, const dimensions_t &gridDimensions, bool periodic) {
//			/** Limiting tempPosition to be within the grid range */
//			if(periodic) {
//				pointPosition.x = roundClamp<Scalar>(pointPosition.x, 0, static_cast<Scalar> (gridDimensions.x) - 1e-3f);
//				pointPosition.y = clamp<Scalar>(pointPosition.y, 0, static_cast<Scalar> (gridDimensions.y) - (1 + 1e-3f));
//			} else {
//				pointPosition.x = clamp<Scalar>(pointPosition.x, 0, static_cast<Scalar> (gridDimensions.x) - (1 + 1e-3f));
//				pointPosition.y = clamp<Scalar>(pointPosition.y, 0, static_cast<Scalar> (gridDimensions.y) - (1 + 1e-3f));
//			}
//			
//			i = static_cast<int> (floor(pointPosition.x)); 
//			j = static_cast<int> (floor(pointPosition.y));
//		}
//
//		FORCE_INLINE void limitToGridRange(Vector3 &pointPosition, int &i, int &j, int &k, const dimensions_t &gridDimensions, bool periodic) {
//			/** Limiting tempPosition to be within the grid range */
//			if(periodic) {
//				pointPosition.x = roundClamp<Scalar>(pointPosition.x, 0, static_cast<Scalar> (gridDimensions.x));
//				pointPosition.y = clamp<Scalar>(pointPosition.y, 0, static_cast<Scalar> (gridDimensions.y) - 1e-3f);
//				pointPosition.z = roundClamp<Scalar>(pointPosition.z, 0, static_cast<Scalar> (gridDimensions.z));
//
//			} else {
//				pointPosition.x = clamp<Scalar>(pointPosition.x, 0, static_cast<Scalar> (gridDimensions.x) - 1e-3f);
//				pointPosition.y = clamp<Scalar>(pointPosition.y, 0, static_cast<Scalar> (gridDimensions.y) - 1e-3f);
//				pointPosition.z = clamp<Scalar>(pointPosition.z, 0, static_cast<Scalar> (gridDimensions.z) - 1e-3f);
//			}
//			i = static_cast<int> (floor(pointPosition.x));
//			j = static_cast<int> (floor(pointPosition.y)); 
//			k = static_cast<int> (floor(pointPosition.z));
//		}
//
//		
//		/************************************************************************/
//		/* Integration functions 2D - Interpolation                             */
//		/************************************************************************/
//		/** The midpoint Runge-Kutta method. */
//		Vector2 rungeKutta2(trajectoryIntegratorParams_t<Vector2> &params);
//		/** Adaptive RK method: breaks the original time-step into smaller steps in order to enforce CFL ~= 1
//		 ** when calculating back-trajectories. */
//		Vector2 adaptiveRungeKutta(trajectoryIntegratorParams_t<Vector2> &params);
//
//		/** Special cells Runge Kutta 2. */
//		Vector2 rungeKutta2SP(trajectoryIntegratorParams_t<Vector2> &params);
//
//		/** The fourth order Runge-Kutta method.*/
//		Vector2 rungeKutta4(trajectoryIntegratorParams_t<Vector2> &params);
//
//		/************************************************************************/
//		/* Switchers				                                            */
//		/************************************************************************/
//		Vector2 integrateTrajectory(trajectoryIntegratorParams_t<Vector2> &params);
//
//		
//		/************************************************************************/
//		/* Integration functions 3D                                             */
//		/************************************************************************/
//		/** The midpoint Runge-Kutta method. */
//		Vector3 rungeKutta2(trajectoryIntegratorParams_t<Vector3> &params);
//		/** Adaptive RK method: breaks the original time-step into smaller steps in order to enforce CFL ~= 1
//		 ** when calculating back-trajectories. */
//		Vector3 adaptiveRungeKutta(trajectoryIntegratorParams_t<Vector3> &params);
//
//		/************************************************************************/
//		/* Switchers				                                            */
//		/************************************************************************/
//		Vector3 integrateTrajectory(trajectoryIntegratorParams_t<Vector3> &params);
//
//	}
//}
//
//#endif