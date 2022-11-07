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
//#include "Integration/TrajectoryIntegrators.h"
//#include "Utils/SpaceTransforms.h"
//#include "Interpolation/LinearInterpolation2D.h"
//#include "Interpolation/LinearInterpolation3D.h"
//
////Data
//#include "Grids/CutCells2D.h"
//#include "Grids/CutCells3D.h"
//#include "Mesh/TriangleMesh3D.h"
//
//#include "Physics/FlowSolverParameters.h"
//
//namespace Chimera {
//	namespace Advection {
//
//		/************************************************************************/
//		/* Integration 2D		                                                */
//		/************************************************************************/
//		/** The midpoint Runge-Kutta method. */
//		Vector2 rungeKutta2(trajectoryIntegratorParams_t<Vector2> &params) {
//			int i = static_cast <int> (floor(params.initialPosition.x));
//			int j = static_cast <int> (floor(params.initialPosition.y));
//
//			dimensions_t gridDimensions = params.pVelocityField2D->getDimensions();
//			Vector2 v1;
//
//			/** Transforming velocity */
//			if(params.transformVelocity)
//				v1 = transformToCoordinateSystem(i, j, params.initialVelocity, *params.pTransformationMatrices2D)/(*params.pScaleFactors2D)(i, j);
//			else
//				v1 = params.initialVelocity/(*params.pScaleFactors2D)(i, j);
//
//			/** Integrating trajectory */
//			Vector2 tempPosition;
//			tempPosition = params.initialPosition - v1*params.dt*0.5;
//
//			/** Limiting tempPosition to be within the grid range */
//			limitToGridRange(tempPosition, i, j, gridDimensions, params.periodicDomain);
//			
//			/** Interpolating intermediary velocity */
//			if(params.backwardsIntegration) 
//				v1 = bilinearInterpolation(tempPosition, *params.pAuxVelocityField2D, params.periodicDomain);
//			else
//				v1 = bilinearInterpolation(tempPosition, *params.pVelocityField2D, params.periodicDomain);
//
//			/** Transforming velocity */
//			if(params.transformVelocity)
//				v1 = transformToCoordinateSystem(i, j, v1, *params.pTransformationMatrices2D)/(*params.pScaleFactors2D)(i, j);
//			else
//				v1 = v1/(*params.pScaleFactors2D)(i, j);
//
//			/** Integrating with the interpolated velocity calculated in previous step */
//			tempPosition = params.initialPosition - v1*params.dt;
//
//			//** Limiting tempPosition to be within the grid range */
//			limitToGridRange(tempPosition, i, j, gridDimensions, params.periodicDomain);
//
//			/** Interpolating the velocity at the final integrated point*/
//			if(params.backwardsIntegration) 
//				v1 = bilinearInterpolation(tempPosition, *params.pAuxVelocityField2D, params.periodicDomain);
//			else
//				v1 = bilinearInterpolation(tempPosition, *params.pVelocityField2D, params.periodicDomain);
//	
//			/** Saving the final integrated point on the initial position - Used for limit calculation*/
//			params.initialPosition = tempPosition;
//
//			return v1;
//		}
//
//		Vector2 rungeKutta2SP(trajectoryIntegratorParams_t<Vector2> &params) {
//			int i = static_cast <int> (floor(params.initialPosition.x));
//			int j = static_cast <int> (floor(params.initialPosition.y));
//
//			dimensions_t gridDimensions = params.pVelocityField2D->getDimensions();
//			Vector2 v1;
//
//			/** Transforming velocity */
//			if(params.transformVelocity)
//				v1 = transformToCoordinateSystem(i, j, params.initialVelocity, *params.pTransformationMatrices2D)/(*params.pScaleFactors2D)(i, j);
//			else
//				v1 = params.initialVelocity/(*params.pScaleFactors2D)(i, j);
//
//			/** Integrating trajectory */
//			Vector2 tempPosition;
//			tempPosition = params.initialPosition - v1*params.dt*0.5;
//
//			/** Limiting tempPosition to be within the grid range */
//			limitToGridRange(tempPosition, i, j, gridDimensions, params.periodicDomain);
//
//			/** Interpolating intermediary velocity */
//			if(params.backwardsIntegration)
//				v1 = bilinearInterpolation(tempPosition, *params.pAuxVelocityField2D, params.pCutCells2D); 
//			else
//				v1 = bilinearInterpolation(tempPosition, *params.pAuxVelocityField2D, params.pNodeVelocityField2D, params.pCutCells2D);
//
//			/** Transforming velocity */
//			if(params.transformVelocity)
//				v1 = transformToCoordinateSystem(i, j, v1, *params.pTransformationMatrices2D)/(*params.pScaleFactors2D)(i, j);
//			else
//				v1 = v1/(*params.pScaleFactors2D)(i, j);
//
//			/** Integrating with the interpolated velocity calculated in previous step */
//			tempPosition = params.initialPosition - v1*params.dt;
//
//			//** Limiting tempPosition to be within the grid range */
//			limitToGridRange(tempPosition, i, j, gridDimensions, params.periodicDomain);
//
//			/** Interpolating the velocity at the final integrated point*/
//			if(params.backwardsIntegration) 
//				v1 = bilinearInterpolation(tempPosition, *params.pAuxVelocityField2D, params.pCutCells2D);
//			else
//				v1 = bilinearInterpolation(tempPosition, *params.pAuxVelocityField2D, params.pNodeVelocityField2D, params.pCutCells2D);
//
//			/** Saving the final integrated point on the initial position - Used for limit calculation*/
//			params.initialPosition = tempPosition;
//
//			return v1;
//		}
//		/** Adaptive RK method: breaks the original time-step into smaller steps in order to enforce CFL ~= 1
//		 ** when calculating back-trajectories. */
//		Vector2 adaptiveRungeKutta(trajectoryIntegratorParams_t<Vector2> &params) {
//				Vector2 localVel, cflVec;
//				Vector2 tempParticlePos = params.initialPosition;
//				Vector2 lastPosition = params.initialPosition;
//				Vector2 finalVel = params.initialVelocity;
//
//				dimensions_t gridDimensions = params.pVelocityField2D->getDimensions();
//
//				int i, j;
//				Scalar dt = params.dt;
//				bool backwardStep = dt < 0;
//				while((dt > 0 && !backwardStep)|| (backwardStep && dt < 0)) { // Perform a RK-2 constraining CFL ~= 1
//					
//					i = static_cast <int> (floor(tempParticlePos.x));
//					j = static_cast <int> (floor(tempParticlePos.y));
//
//					/** Transforming velocity */
//					if(params.transformVelocity)
//						localVel = transformToCoordinateSystem(i, j, finalVel, *params.pTransformationMatrices2D);
//					else
//						localVel = finalVel;
//
//					Vector2 tempDist = localVel*dt;
//					Vector2 scaleFactor = (*params.pScaleFactors2D)(i, j);
//
//					/** Calculating maximum permitted time step*/
//					if(localVel.length() > 0)
//						cflVec = scaleFactor/tempDist;
//					else
//						return Vector2(0, 0);
//
//					/** Adaptive time step is the smallest of the ratios calculated in cflVec. These ratios are normalized 
//					 ** within the current time step, which means that they vary between 0 and 1.*/
//					Scalar adaptiveTimestep = min(abs(cflVec.x), abs(cflVec.y));
//
//					/** Check outs if a time step can be greater than it already is. If its greater, clamps, since we 
//					 ** do not want a step greater than the leg ;) */
//					if(adaptiveTimestep < 1 && adaptiveTimestep > 0)
//						adaptiveTimestep *= dt;
//					else 
//						adaptiveTimestep = abs(dt);
//
//					/** Scaling the velocity accordingly*/
//					localVel /= (*params.pScaleFactors2D)(i, j);
//					
//					/** Checks out if a backwards integration is being performed. For adaptive runge-kutta this will be
//					 ** treated differently */
//					if(backwardStep)
//						tempParticlePos = lastPosition + localVel*adaptiveTimestep*0.5;
//					else
//						tempParticlePos = lastPosition - localVel*adaptiveTimestep*0.5;
//
//					//** Limiting tempPosition to be within the grid range */
//					limitToGridRange(tempParticlePos, i, j, gridDimensions, params.periodicDomain);
//
//					/** Interpolating intermediary velocity */
//					if(backwardStep)
//						localVel = bilinearInterpolation(tempParticlePos, *params.pAuxVelocityField2D, params.periodicDomain);
//					else 
//						localVel = bilinearInterpolation(tempParticlePos, *params.pVelocityField2D, params.periodicDomain);
//
//					/** Transforming velocity */
//					if(params.transformVelocity)
//						localVel = transformToCoordinateSystem(i, j, localVel, *params.pTransformationMatrices2D)/(*params.pScaleFactors2D)(i, j);
//					else
//						localVel = localVel/(*params.pScaleFactors2D)(i, j);
//
//					/** Checks out if a backwards integration is being performed. For adaptive runge-kutta this will be
//					 ** treated differently */
//					if(backwardStep)
//						tempParticlePos = lastPosition + localVel*adaptiveTimestep;
//					else
//						tempParticlePos = lastPosition - localVel*adaptiveTimestep;
//
//					//** Limiting tempPosition to be within the grid range */
//					limitToGridRange(tempParticlePos, i, j, gridDimensions, params.periodicDomain);
//
//					/** Interpolating intermediary velocity */					
//					if(params.transformVelocity)
//						localVel = bilinearInterpolation(tempParticlePos, *params.pAuxVelocityField2D, params.periodicDomain);
//					else 
//						localVel = bilinearInterpolation(tempParticlePos, *params.pVelocityField2D, params.periodicDomain);
//
//					/** Updating next time step */
//					if(backwardStep)
//						dt += adaptiveTimestep;
//					else
//						dt -= adaptiveTimestep;
//
//					lastPosition = tempParticlePos;
//				}
//				params.initialPosition = lastPosition;
//				return finalVel;
//		}
//
//		Vector2 rungeKutta4(trajectoryIntegratorParams_t<Vector2> &params) {
//			int i = static_cast <int> (floor(params.initialPosition.x));
//			int j = static_cast <int> (floor(params.initialPosition.y));
//
//			dimensions_t gridDimensions = params.pVelocityField2D->getDimensions();
//			Vector2 v1, v2, v3, v4;
//
//			/************************************************************************/
//			/* v1 calculation (initial given velocity)                              */
//			/************************************************************************/
//			/** Transforming velocity */
//			if(params.transformVelocity)
//				v1 = transformToCoordinateSystem(i, j, params.initialVelocity, *params.pTransformationMatrices2D)/(*params.pScaleFactors2D)(i, j);
//			else
//				v1 = params.initialVelocity/(*params.pScaleFactors2D)(i, j);
//
//			/** Integrating trajectory */
//			Vector2 tempPosition;
//			tempPosition = params.initialPosition - v1*params.dt*0.5;
//			
//			/************************************************************************/
//			/* v2 calculation (updated with h = 0.5)                                */
//			/************************************************************************/
//			/** Interpolating intermediary velocity */
//			if(params.backwardsIntegration) 
//				v2 = bilinearInterpolation(tempPosition, *params.pAuxVelocityField2D, params.periodicDomain);
//			else
//				v2 = bilinearInterpolation(tempPosition, *params.pVelocityField2D, params.periodicDomain);
//
//			/** Transforming velocity */
//			if(params.transformVelocity)
//				v2 = transformToCoordinateSystem(i, j, v2, *params.pTransformationMatrices2D)/(*params.pScaleFactors2D)(i, j);
//			else
//				v2 = v2/(*params.pScaleFactors2D)(i, j);
//
//			/** Integrating with the interpolated velocity calculated in previous step */
//			tempPosition = params.initialPosition - v2*params.dt*0.5;
//
//			/************************************************************************/
//			/* v3 calculation (updated with h = 0.5)                                */
//			/************************************************************************/
//			/** Interpolating intermediary velocity */
//			if(params.backwardsIntegration) 
//				v3 = bilinearInterpolation(tempPosition, *params.pAuxVelocityField2D, params.periodicDomain);
//			else
//				v3 = bilinearInterpolation(tempPosition, *params.pVelocityField2D, params.periodicDomain);
//
//			/** Transforming velocity */
//			if(params.transformVelocity)
//				v3 = transformToCoordinateSystem(i, j, v3, *params.pTransformationMatrices2D)/(*params.pScaleFactors2D)(i, j);
//			else
//				v3 = v3/(*params.pScaleFactors2D)(i, j);
//
//			/** Integrating with the interpolated velocity calculated in previous step */
//			tempPosition = params.initialPosition - v3*params.dt;
//
//			/************************************************************************/
//			/* v4 calculation (updated with h = 1.0)                                */
//			/************************************************************************/
//			/** Interpolating intermediary velocity */
//			if(params.backwardsIntegration) 
//				v4 = bilinearInterpolation(tempPosition, *params.pAuxVelocityField2D, params.periodicDomain);
//			else
//				v4 = bilinearInterpolation(tempPosition, *params.pVelocityField2D, params.periodicDomain);
//
//			/** Transforming velocity */
//			if(params.transformVelocity)
//				v4 = transformToCoordinateSystem(i, j, v4, *params.pTransformationMatrices2D)/(*params.pScaleFactors2D)(i, j);
//			else
//				v4 = v4/(*params.pScaleFactors2D)(i, j);
//
//			/************************************************************************/
//			/* Final position integration											*/
//			/************************************************************************/
//			tempPosition =  params.initialPosition - (v1 + v2 + v3 + v4)*(1/6.0f)*params.dt;
//
//			/** Saving the final integrated point on the initial position*/
//			params.initialPosition = tempPosition;
//			if(params.backwardsIntegration) 
//				v1 = bilinearInterpolation(tempPosition, *params.pAuxVelocityField2D, params.periodicDomain);
//			else
//				v1 = bilinearInterpolation(tempPosition, *params.pVelocityField2D, params.periodicDomain);
//	
//			return v1;
//		}
//
//
//		/************************************************************************/
//		/* Switchers				                                            */
//		/************************************************************************/
//		Vector2 integrateTrajectory(trajectoryIntegratorParams_t<Vector2> &params) {
//			switch(params.integrationMethod) {
//			case RungeKutta_2:
//				if(params.pCutCells2D != NULL)
//					return rungeKutta2SP(params);
//				else
//					return rungeKutta2(params);
//				break;
//
//			case RungeKutta_Adaptive:
//				return adaptiveRungeKutta(params);
//				break;
//
//			default:
//				return rungeKutta2(params);
//				break;
//			}
//		}
//
//		/************************************************************************/
//		/* Integration 3D		                                                */
//		/************************************************************************/
//		
//		Vector3 rungeKutta2(trajectoryIntegratorParams_t<Vector3> &params){
//				dimensions_t gridDimensions = params.pVelocityField3D->getDimensions();
//
//				int i = static_cast <int> (floor(params.initialPosition.x)); 
//				int j = static_cast <int> (floor(params.initialPosition.y)); 
//				int k = static_cast <int> (floor(params.initialPosition.z));
//				Vector3 v1;
//
//				if(params.transformVelocity)
//					v1 = transformToCoordinateSystem(i, j, k, params.initialVelocity, *params.pTransformationMatrices3D)/(*params.pScaleFactors3D)(i, j, k);
//				else
//					v1 = params.initialVelocity/(*params.pScaleFactors3D)(i, j, k);
//
//				Vector3 tempPosition = params.initialPosition - v1*params.dt*0.5;
//
//				limitToGridRange(tempPosition, i, j, k, gridDimensions, params.periodicDomain);
//
//				v1 = trilinearInterpolation(tempPosition, *params.pVelocityField3D, params.periodicDomain);
//
//				if(params.transformVelocity)
//					v1 = transformToCoordinateSystem(i, j, k, v1, *params.pTransformationMatrices3D)/(*params.pScaleFactors3D)(i, j, k);
//				else
//					v1 = v1/(*params.pScaleFactors3D)(i, j, k);
//
//				tempPosition = params.initialPosition - v1*params.dt;
//
//				limitToGridRange(tempPosition, i, j, k, gridDimensions, params.periodicDomain);
//
//				v1 = trilinearInterpolation(tempPosition, *params.pVelocityField3D, params.periodicDomain);
//				
//				params.initialPosition = tempPosition;
//				return v1;
//		}
//		Vector3 rungeKutta2SP(trajectoryIntegratorParams_t<Vector3> &params) {			
//			dimensions_t gridDimensions = params.pNodeVelocityField3D->pGridNodesVelocities->getDimensions();
//
//			int i = static_cast <int> (floor(params.initialPosition.x)); 
//			int j = static_cast <int> (floor(params.initialPosition.y)); 
//			int k = static_cast <int> (floor(params.initialPosition.z));
//			Vector3 v1;
//
//			Scalar dx = params.pCutCells3D->getGridSpacing();
//			
//			if(params.transformVelocity)
//				v1 = transformToCoordinateSystem(i, j, k, params.initialVelocity, *params.pTransformationMatrices3D)/dx;
//			else
//				v1 = params.initialVelocity/dx;
//
//			Vector3 tempPosition = params.initialPosition - v1*params.dt*0.5;
//
//			limitToGridRange(tempPosition, i, j, k, gridDimensions, params.periodicDomain);
//
//			//v1 = trilinearInterpolation(tempPosition, params.pCutCells3D, params.pNodeVelocityField3D, Math::sbcInterpolation);
//
//			if(params.transformVelocity)
//				v1 = transformToCoordinateSystem(i, j, k, v1, *params.pTransformationMatrices3D)/dx;
//			else
//				v1 = v1/dx;
//
//			tempPosition = params.initialPosition - v1*params.dt;
//
//			limitToGridRange(tempPosition, i, j, k, gridDimensions, params.periodicDomain);
//
//			//v1 = trilinearInterpolation(tempPosition, params.pCutCells3D, params.pNodeVelocityField3D, Math::sbcInterpolation);
//
//			params.initialPosition = tempPosition;
//			return v1;
//		}
//
//		Vector3 adaptiveRungeKutta(trajectoryIntegratorParams_t<Vector3> &params) {
//				dimensions_t gridDimensions = params.pVelocityField2D->getDimensions();
//
//				Vector3 localVel, cflVec;
//				Vector3 tempParticlePos = params.initialPosition;
//				Vector3 lastPosition = params.initialPosition;
//				Vector3 finalVel = params.initialVelocity;
//				int i, j, k;
//
//				Scalar dt = params.dt;
//				bool backwardStep = dt < 0;
//
//				while((dt > 0 && !backwardStep)|| (backwardStep && dt < 0)) { // Perform a RK-2 constraining CFL ~= 1
//					limitToGridRange(tempParticlePos, i, j, k, gridDimensions, params.periodicDomain);
//
//					if(params.transformVelocity)
//						localVel = transformToCoordinateSystem(i, j, k, finalVel, *params.pTransformationMatrices3D);
//					else
//						localVel = finalVel;
//
//					if(localVel.length() > 0)
//						cflVec = ((*params.pScaleFactors3D)(i, j, k)*(1 + 1e-2f))/localVel.length();
//					else
//						return Vector3(0, 0, 0);
//
//					Scalar adaptiveTimestep = max(cflVec.z, max(abs(cflVec.x), abs(cflVec.y)));
//
//					if(adaptiveTimestep > dt) //Lets not do a step greater than the leg ;)
//						adaptiveTimestep = abs(dt);
//
//					localVel /= (*params.pScaleFactors3D)(i, j, k);
//
//					//Backwards integration
//					if(backwardStep)
//						tempParticlePos = lastPosition + localVel*adaptiveTimestep*0.5;
//					else
//						tempParticlePos = lastPosition - localVel*adaptiveTimestep*0.5;
//
//					localVel = trilinearInterpolation(tempParticlePos, *params.pVelocityField3D, params.periodicDomain);
//
//					if(params.transformVelocity)
//						localVel = transformToCoordinateSystem(i, j, k, localVel, *params.pTransformationMatrices3D)/(*params.pScaleFactors3D)(i, j, k);
//					else
//						localVel = localVel/(*params.pScaleFactors3D)(i, j, k);;
//
//					//Backwards integration
//					if(backwardStep)
//						tempParticlePos = lastPosition + localVel*adaptiveTimestep;
//					else
//						tempParticlePos = lastPosition - localVel*adaptiveTimestep;
//
//					finalVel = trilinearInterpolation(tempParticlePos, *params.pVelocityField3D, params.periodicDomain);
//
//					if(backwardStep)
//						dt += adaptiveTimestep;
//					else
//						dt -= adaptiveTimestep;
//
//					lastPosition = tempParticlePos;
//				}
//				params.initialPosition = lastPosition;
//				return finalVel;
//		}
//
//
//		/************************************************************************/
//		/* Switchers				                                            */
//		/************************************************************************/
//		Vector3 integrateTrajectory(trajectoryIntegratorParams_t<Vector3> &params) {
//			switch(params.integrationMethod) {
//			case RungeKutta_2:
//				if(params.pCutCells3D != NULL)
//					return rungeKutta2SP(params);
//				else
//					return rungeKutta2(params);
//				break;
//
//			case RungeKutta_Adaptive:
//				return adaptiveRungeKutta(params);
//				break;
//
//			default:
//				return rungeKutta2(params);
//				break;
//			}
//		}
//	}
//}
