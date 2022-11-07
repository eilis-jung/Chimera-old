//#ifndef _CHIMERA_PRE_SIMULATION_APP_H_
//#define _CHIMERA_PRE_SIMULATION_APP_H_
//#pragma  once
//
//#include "Applications/Application3D.h"
//
//
//
//namespace Chimera {
//
//
//	class PreSimulation : public Application3D {
//
//	public:
//
//		/************************************************************************/
//		/* ctors                                                                */
//		/************************************************************************/
//		PreSimulation(int argc, char** argv, TiXmlElement *pChimeraConfig);
//
//
//		/************************************************************************/
//		/* Functionalities                                                      */
//		/************************************************************************/
//		void draw();
//		void update();
//
//		/************************************************************************/
//		/* Callbacks                                                            */
//		/************************************************************************/
//		void keyboardCallback(unsigned char key, int x, int y) {
//			m_pRenderer->keyboardCallback(key, x, y);
//			switch(key) {
//				case '+':
//					m_pSceneLoader->getCar()->getPhysXVehicle()->gearUp();
//				break;
//
//				case 'r':
//					m_pSceneLoader->getCar()->getPhysXVehicle()->applyRandomForce();
//				break;
//
//				case '1':
//					m_pRenderer->getCamera()->followObject(m_pSceneLoader->getCar(), Vector3(1, 0.5, 0));
//				break;
//			}
//		}
//
//		void specialKeyboardCallback(int key, int x, int y) {
//			switch(key) {
//				case GLUT_KEY_UP:
//					m_pSceneLoader->getCar()->accelerate(1.0f);
//				break;
//				case GLUT_KEY_DOWN:
//					m_pSceneLoader->getCar()->accelerate(-1.0f);
//				break;
//				case GLUT_KEY_LEFT:
//					m_pSceneLoader->getCar()->steer(-1.0f);
//				break;
//				case GLUT_KEY_RIGHT:
//					m_pSceneLoader->getCar()->steer(1.0f);
//				break;
//			}
//		}
//
//		void specialKeyboardUpCallback(int key, int x, int y) {
//			switch(key) {
//				case GLUT_KEY_UP:
//					m_pSceneLoader->getCar()->accelerate(0.0f);
//				break;
//				case GLUT_KEY_DOWN:
//					m_pSceneLoader->getCar()->accelerate(0.0f);
//				break;
//				case GLUT_KEY_LEFT:
//					m_pSceneLoader->getCar()->steer(0.0f);
//				break;
//				case GLUT_KEY_RIGHT:
//					m_pSceneLoader->getCar()->steer(0.0f);
//				break;
//			}
//		}
//
//		virtual void exitCallback() {
//			logOutputStream->seekp(0, ios_base::beg);
//			*logOutputStream << m_currentFrame << " " << m_pPhysicsCore->getElapsedTime() << endl;
//		}
//		
//		
//	private:
//
//		void logCarPosition();
//
//		/************************************************************************/
//		/* Loading functions                                                    */
//		/************************************************************************/
//		void loadPhysics();
//
//		/************************************************************************/
//		/* Class members                                                        */
//		/************************************************************************/
//		string m_logFilename;
//		auto_ptr<ofstream> logOutputStream;
//		int m_currentFrame;
//		shared_ptr<PhongShader> m_pPhongShading;
//	};
//}
//
//
//#endif