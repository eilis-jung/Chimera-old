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
//
//#include "Primitives/Light.h"
//
//namespace Chimera {
//
//
//	Light::Light(GLenum lightID, const Vector3 &position) :
//			PhysicalObject(position, Vector3(0, 0, 0), Vector3(1, 1, 1)), 
//			m_lightID(lightID) {
//		float specular[] = {1.0, 1.0, 1.0, 1.0};
//		float diffuse[] = {1.0f, 1.0f, 1.0, 1.0};
//		float ambient[] = {0.2f, 0.2f, 0.2f, 0.2f};
//		float glPosition[] = { m_position.x, m_position.y, m_position.z, 0.0f };
//
//		glLightfv(m_lightID, GL_AMBIENT, ambient);
//		glLightfv(m_lightID, GL_SPECULAR, specular);
//		glLightfv(m_lightID, GL_DIFFUSE, diffuse);
//		glLightfv(m_lightID, GL_POSITION, glPosition);
//		glEnable(m_lightID);
//	}
//	void Light::draw() {
//		glDisable(GL_LIGHTING);
//		glDisable(GL_TEXTURE_2D);
//		glPushMatrix();
//			
//			glTranslatef(m_position.x, m_position.y, m_position.z);
//			glutSolidSphere(1.0f, 100, 10);
//		
//		glPopMatrix();
//		glEnable(GL_TEXTURE_2D);
//		glEnable(GL_LIGHTING);
//	}
//}
