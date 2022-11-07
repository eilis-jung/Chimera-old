//#include "Integration/UpwindGradientIntegrator.h"
//
//namespace Chimera {
//
//	namespace Advection {
//
//		template<>
//		void UpwindGradientIntegrator<Vector2>::integrateScalarField(Array<Scalar> *pScalarField,  Scalar dx, Scalar dt) {
//			Array2D<Scalar> *pLsArray = dynamic_cast<Array2D<Scalar> *>(pScalarField);
//			Array2D<Scalar> tempArray  = *pLsArray;
//
//			for(int i = 1; i < pLsArray->getDimensions().x - 1; i++) {
//				for(int j = 1; j < pLsArray->getDimensions().y - 1; j++) {
//					Scalar diXm = (tempArray(i, j) - tempArray(i - 1, j))/dx; //Di_x -
//					Scalar diXp = (tempArray(i + 1, j) - tempArray(i, j))/dx; //Di_x +
//
//					Scalar diYm = (tempArray(i, j) - tempArray(i, j - 1))/dx; //Di_x -
//					Scalar diYp = (tempArray(i, j + 1) - tempArray(i, j))/dx; //Di_x +
//
//					Scalar di = sqrt(pow(max(diXm, 0.0f), 2) + pow(min(diXp, 0.0f), 2) +
//						pow(max(diYm, 0.0f), 2) + pow(min(diYp, 0.0f), 2)); 
//					
//				/*	Scalar curvature = -calculateCurvature(i, j, tempArray, dx);
//					Scalar velocity = (1 - 15*curvature);*/
//
//					Vector2 lsGradient = interpolateGradient(Vector2(i + 0.5f, j + 0.5f) , tempArray);
//					Scalar finalVelocity = lsGradient.length();
//
//					Scalar lsValue = tempArray(i, j) - dt*(di - finalVelocity);
//					(*pLsArray)(i, j) = lsValue; 
//				}
//			}
//
//			//Enforcing Neumann boundary conditions
//			for(int i = 0; i < pLsArray->getDimensions().x; i++) {
//				(*pLsArray)(i, 1) = (*pLsArray)(i, 0);
//				(*pLsArray)(i, pLsArray->getDimensions().y - 2) = (*pLsArray)(i, pLsArray->getDimensions().y - 1);
//			}
//			for(int j = 0; j < pLsArray->getDimensions().y; j++) {
//				(*pLsArray)(1, j) = (*pLsArray)(0, j);
//				(*pLsArray)(pLsArray->getDimensions().x - 2, 1) = (*pLsArray)(pLsArray->getDimensions().x - 1, j);
//			}
//		}
//
//		template<>
//		void UpwindGradientIntegrator<Vector3>::integrateScalarField(Array<Scalar> *pScalarField,  Scalar dx, Scalar dt) {
//			
//		}
//	}
//}