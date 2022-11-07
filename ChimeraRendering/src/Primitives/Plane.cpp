//#include "Primitives/Plane.h"
//#include "Physics/PhysicsCore.h"
//
//namespace Chimera {
//	namespace Rendering {
//
//		#pragma region Constructors
//		Plane::Plane(const Vector3 & position, const Vector3 &upVec, const Vector2 &planeSize, Scalar tilingSpacing) : 
//			Object3D(position, Vector3(0, 0, 0), Vector3(1, 1, 1)), m_mesh(position, upVec, planeSize, tilingSpacing) {
//			m_drawPoints = false;
//		}
//		#pragma endregion Constructors
//		
//		#pragma region Functionalities
//		void Plane::draw() {
//			Array2D<Vector3> *pPoints = m_mesh.getPointsPtr();
//			if(m_drawPoints) {
//				glBegin(GL_POINTS);
//				for(int i = 0; i < pPoints->getDimensions().x; i++) {
//					for(int j = 0; j < pPoints->getDimensions().y ; j++) {
//						glVertex3f((*pPoints)(i, j).x, (*pPoints)(i, j).y, (*pPoints)(i, j).z); 
//					}
//				}
//				glEnd();
//			}
//			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//			glColor3f(0.945f, 0.713f, 0.180f);
//			//glColor3f(0.2f, 0.33f, 1.0f);
//			glBegin(GL_QUADS);
//			for(int i = 0; i < pPoints->getDimensions().x - 1; i++) {
//				for(int j = 0; j < pPoints->getDimensions().y - 1; j++) {
//					glVertex3f((*pPoints)(i, j).x, (*pPoints)(i, j).y, (*pPoints)(i, j).z); 
//					glVertex3f((*pPoints)(i + 1, j).x, (*pPoints)(i + 1, j).y, (*pPoints)(i + 1, j).z);
//					glVertex3f((*pPoints)(i + 1, j + 1).x, (*pPoints)(i + 1, j + 1).y, (*pPoints)(i + 1, j + 1).z); 
//					glVertex3f((*pPoints)(i, j + 1).x, (*pPoints)(i, j + 1).y, (*pPoints)(i, j + 1).z); 
//				}
//			}
//			glEnd();
//			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//		}
//
//		void Plane::update(Scalar dt) {
//			Vector3 positionOffset = updateVelocityFunction(PhysicsCore::getInstance()->getElapsedTime() + dt);
//			for (int i = 0; i < m_mesh.getPointsPtr()->getDimensions().x; i++) {
//				for (int j = 0; j < m_mesh.getPointsPtr()->getDimensions().y; j++) {
//					(*m_mesh.getPointsPtr())(i, j) = m_mesh.getInitialPoints()(i, j) + positionOffset;
//				}
//			}
//		}
//		#pragma endregion Functionalities
//	}
//}