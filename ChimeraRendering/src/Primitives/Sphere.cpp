//#include "Primitives/Sphere.h"
//
//namespace Chimera {
//	namespace Rendering {
//
//		#pragma region Constructors
//		Sphere::Sphere(const Vector3 &position, Scalar radius) : Object3D(position, Vector3(0, 0, 0), Vector3(1,1,1)) {
//			m_radius = radius;
//			m_objectName = "Sphere" + intToStr(rand());
//		}
//		#pragma endregion 
//
//		#pragma region Functionalities
//		void Sphere::draw() {
//			glLineWidth(1.0f);
//
//			if (m_drawShaded) {
//				glPushMatrix();
//				glColor3f(m_color[0], m_color[1], m_color[2]);
//				glTranslatef(m_position.x, m_position.y, m_position.z);
//				glutSolidSphere(m_radius, 10, 8);
//				glPopMatrix();
//			}
//			
//			if (m_drawWireframe) {
//				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
//				glPushMatrix();
//				glColor3f(m_color[0], m_color[1], m_color[2]);
//				glTranslatef(m_position.x, m_position.y, m_position.z);
//				glutSolidSphere(m_radius, 10, 8);
//				glPopMatrix();
//				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//			}
//		}
//
//		void Sphere::update(Scalar dt) {
//
//		}
//		#pragma endregion 
//	}
//}