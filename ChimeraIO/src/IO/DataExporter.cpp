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


#include "IO/DataExporter.h"
#include "SDL/SDL_savepng.h"

namespace Chimera {

	#pragma region Constructors
	template <class VectorType, template <class> class ArrayType>
	DataExporter<VectorType, ArrayType>::DataExporter(const configParams_t &params, const dimensions_t &gridDimensions) :
		m_params(params),
		m_densityFields(gridDimensions), m_pressureFields(gridDimensions), m_velocityBuffer(gridDimensions) {
		m_pixels.resize(params.getScreenHeight() * params.getScreenWidth() * 3); // assume that the screen size is fixed
		m_sdlSurface = SDL_CreateRGBSurface(SDL_SWSURFACE, params.getScreenWidth(), params.getScreenHeight(), 24, 0x000000FF, 0x0000FF00, 0x00FF0000, 0);
		m_numDumpedFrames = 0;		
	}

	#pragma region LoggingFunctions
	template <>
	void DataExporter<Vector2, Array2D>::logDumpDensity(Scalar timeElapsed) {

		GridData2D *pGridData = m_pFlowSolver->getGrid()->getGridData2D();
		dimensions_t tempDimensions = m_pFlowSolver->getGrid()->getDimensions();
		//Logging just times that passed the fps
		for (int i = 0; i < tempDimensions.x; i++) {
			for (int j = 0; j < tempDimensions.y; j++) {
				m_densityFields(i, j) = pGridData->getDensityBuffer().getValue(i, j);
			}
		}

		dumpDensity();
	}

	template <>
	void DataExporter<Vector3, Array3D>::logDumpDensity(Scalar timeElapsed) {
		
		GridData3D *pGridData = m_pFlowSolver->getGrid()->getGridData3D();
		dimensions_t tempDimensions = m_pFlowSolver->getGrid()->getDimensions();
		for (int i = 0; i < tempDimensions.x; i++) {
			for (int j = 0; j < tempDimensions.y; j++) {
				for (int k = 0; k < tempDimensions.z; k++) {
					m_densityFields(i, j, k) = pGridData->getDensityBuffer().getValue(i, j, k);
				}
			}
		}
		dumpDensity();
	}

	template<>
	void DataExporter<Vector2, Array2D>::logDumpPressure(Scalar timeElapsed) {
		
		GridData2D *pGridData = m_pFlowSolver->getGrid()->getGridData2D();
		dimensions_t tempDimensions = m_pFlowSolver->getGrid()->getDimensions();
		//Logging just times that passed the fps
		for (int i = 0; i < tempDimensions.x; i++) {
			for (int j = 0; j < tempDimensions.y; j++) {
				m_pressureFields(i, j) = pGridData->getPressure(i, j);
			}
		}

		dumpPressure();
	}

	template <>
	void DataExporter<Vector3, Array3D>::logDumpPressure(Scalar timeElapsed) {
		GridData3D *pGridData = m_pFlowSolver->getGrid()->getGridData3D();
		dimensions_t tempDimensions = m_pFlowSolver->getGrid()->getDimensions();
		//Logging just times that passed the fps
		for (int i = 0; i < tempDimensions.x; i++) {
			for (int j = 0; j < tempDimensions.y; j++) {
				for (int k = 0; k < tempDimensions.z; k++) {
					m_pressureFields(i, j, k) = pGridData->getPressure(i, j, k);
				}
			}
		}
		dumpPressure();
	}

	template<>
	void DataExporter<Vector2, Array2D>::logDumpVelocity(Scalar timeElapsed) {
		
		GridData2D *pGridData = m_pFlowSolver->getGrid()->getGridData2D();
		dimensions_t tempDimensions = m_pFlowSolver->getGrid()->getDimensions();
		//Logging just times that passed the fps
		for (int i = 0; i < tempDimensions.x; i++) {
			for (int j = 0; j < tempDimensions.y; j++) {
				if (m_params.pNodeBasedVelocities) {
					m_velocityBuffer(i, j) = (*m_params.pNodeBasedVelocities)(i, j);
				}
				else {
					m_velocityBuffer(i, j) = pGridData->getVelocity(i, j);
				}
			}
		}

		dumpVelocity();
	}

	template <>
	void DataExporter<Vector3, Array3D>::logDumpVelocity(Scalar timeElapsed) {
		
		GridData3D *pGridData = m_pFlowSolver->getGrid()->getGridData3D();
		dimensions_t tempDimensions = m_pFlowSolver->getGrid()->getDimensions();
		//Logging just times that passed the fps
		for (int i = 0; i < tempDimensions.x; i++) {
			for (int j = 0; j < tempDimensions.y; j++) {
				for (int k = 0; k < tempDimensions.z; k++) {
					if (m_params.pNodeBasedVelocities) {
						m_velocityBuffer(i, j, k) = (*m_params.pNodeBasedVelocities)(i, j, k);
					}
					else {
						m_velocityBuffer(i, j, k) = pGridData->getVelocity(i, j, k);
					}
				}
			}
		}

		dumpVelocity();
	}

	template<>
	void DataExporter<Vector2, Array2D>::logDumpThinObject(Scalar timeElapsed) {
		//for(unsigned int ithCfg = 0; ithCfg < m_objectBufferIndices.size(); ithCfg++) {
		//	for(int i = 0; i < m_pObjectPoints.size(); i++) {
		//		vector<Vector2> *pObjectPoints = m_pObjectPoints[i];
		//		if(timeElapsed > m_objectBufferIndices[ithCfg]*(1/(float) m_params.frameRate)) { //Logging just times that passed the fps
		//			vector<Vector2> *pBuffObjPoints = m_pObjectBuffers[ithCfg];
		//			for(int i = 0; i < pObjectPoints->size(); i++) {
		//				int index = getSimulationIndex(i, m_objectBufferIndices[ithCfg], pObjectPoints->size());
		//				(*pBuffObjPoints)[index] = pObjectPoints->at(i);
		//			}
		//			m_objectBufferIndices[ithCfg] = m_objectBufferIndices[ithCfg] + 1;
		//		}
		//	}
		//}
		//dumpThinObject();
	}

	template<>
	void DataExporter<Vector3, Array3D>::logDumpThinObject(Scalar timeElapsed) {

	}

	
	void DataExporter<Vector2, Array2D>::logDumpSpecialCells(Scalar timeElapsed) {

	}

	void DataExporter<Vector3, Array3D>::logDumpSpecialCells(Scalar timeElapsed) {

	}

	template <class VectorType, template <class> class ArrayType>
	void DataExporter<VectorType, ArrayType>::logSaveScreenshot() {
		// resize
		const int w = m_params.getScreenWidth();
		const int h = m_params.getScreenHeight();		

		// read pixel
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, (GLsizei)w, (GLsizei)h, GL_RGB, GL_UNSIGNED_BYTE, &m_pixels[0]);
		
		string screenShotExportName;
		if (m_velocityBuffer.getDimensions().z == 0) {
			screenShotExportName = "Flow Logs/2D/Screenshot/" + m_params.getScreenshotFilename() + intToStr(m_numDumpedFrames) + ".png";
		} else {
			screenShotExportName = "Flow Logs/3D/Screenshot/" + m_params.getScreenshotFilename() + intToStr(m_numDumpedFrames) + ".png";
		}

		for (int i = 0; i < h; ++i)
			memcpy(((char *)m_sdlSurface->pixels) + m_sdlSurface->pitch * i, &m_pixels[0] + 3 * w* (h - i - 1), w * 3);
		
		SDL_SavePNG(m_sdlSurface, screenShotExportName.c_str());
		
	}
	#pragma endregion

	#pragma region DumpingFunctions
	template<>
	void DataExporter<Vector2, Array2D>::dumpDensity() {
		string densityExportName("Flow Logs/2D/Density/" + m_params.getDensityFilename() + intToStr(m_numDumpedFrames) + ".log");
		
		dimensions_t gridDimensions = m_pFlowSolver->getGrid()->getDimensions();

		auto_ptr<ofstream> fileStream(new ofstream(densityExportName.c_str(), ofstream::binary));
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.x), sizeof(gridDimensions.x));		//int
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.y), sizeof(gridDimensions.y));		//int
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.z), sizeof(gridDimensions.z));		//int

		for (int i = 0; i < gridDimensions.x; i++) {
			for (int j = 0; j < gridDimensions.y; j++) {
				fileStream->write(reinterpret_cast<const char*>(&m_densityFields(i, j)), sizeof(Scalar));
			}
		}
		if (fileStream->is_open())
			Logger::get() << " Density successfully logged and dumped to file: " << densityExportName << endl;

		
	}

	template <>
	void DataExporter<Vector3, Array3D>::dumpDensity() {
		string densityExportName("Flow Logs/3D/Density/" + m_params.getDensityFilename() + intToStr(m_numDumpedFrames) + ".log");
		
		dimensions_t gridDimensions = m_pFlowSolver->getGrid()->getDimensions();

		auto_ptr<ofstream> fileStream(new ofstream(densityExportName.c_str(), ofstream::binary));
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.x), sizeof(gridDimensions.x));		//Scalar
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.y), sizeof(gridDimensions.y));		//Scalar
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.z), sizeof(gridDimensions.z));		//Scalar

		for (int i = 0; i < gridDimensions.x; i++) {
			for (int j = 0; j < gridDimensions.y; j++) {
				for (int k = 0; k < gridDimensions.z; k++) {
					fileStream->write(reinterpret_cast<const char*>(&m_densityFields(i, j, k)), sizeof(Scalar));
				}
			}
		}

		Logger::get() << " Density successfully logged and dumped to file: " << densityExportName << endl;
	}


	template<>
	void DataExporter<Vector2, Array2D>::dumpPressure() {
		string pressureExportName("Flow Logs/2D/Pressure/" + m_params.getPressureFilename() + intToStr(m_numDumpedFrames) + ".log");
		
		dimensions_t gridDimensions = m_pFlowSolver->getGrid()->getDimensions();

		auto_ptr<ofstream> fileStream(new ofstream(pressureExportName.c_str(), ofstream::binary));
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.x), sizeof(gridDimensions.x));		//int
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.y), sizeof(gridDimensions.y));		//int
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.z), sizeof(gridDimensions.z));		//int

		for (int i = 0; i < gridDimensions.x; i++) {
			for (int j = 0; j < gridDimensions.y; j++) {
				fileStream->write(reinterpret_cast<const char*>(&m_pressureFields(i, j)), sizeof(Scalar));
			}
		}
		if (fileStream->is_open())
			Logger::get() << " Pressure successfully logged and dumped to file: " << pressureExportName << endl;

	}

	template <>
	void DataExporter<Vector3, Array3D>::dumpPressure() {
		string pressureExportName("Flow Logs/3D/Density/" + m_params.getPressureFilename() + intToStr(m_numDumpedFrames) + ".log");
		
		dimensions_t gridDimensions = m_pFlowSolver->getGrid()->getDimensions();

		auto_ptr<ofstream> fileStream(new ofstream(pressureExportName.c_str(), ofstream::binary));
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.x), sizeof(gridDimensions.x));		//Scalar
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.y), sizeof(gridDimensions.y));		//Scalar
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.z), sizeof(gridDimensions.z));		//Scalar

		for (int i = 0; i < gridDimensions.x; i++) {
			for (int j = 0; j < gridDimensions.y; j++) {
				for (int k = 0; k < gridDimensions.z; k++) {
					fileStream->write(reinterpret_cast<const char*>(&m_pressureFields(i, j, k)), sizeof(Scalar));
				}
			}
		}

		Logger::get() << " Pressure successfully logged and dumped to file: " << pressureExportName << endl;
	}


	template<>
	void DataExporter<Vector2, Array2D>::dumpVelocity() {
		string velocityExportName("Flow Logs/2D/Velocity/" + m_params.getVelocityFilename() + intToStr(m_numDumpedFrames) + ".log");
		
		dimensions_t gridDimensions = m_pFlowSolver->getGrid()->getDimensions();

		auto_ptr<ofstream> fileStream(new ofstream(velocityExportName.c_str(), ofstream::binary));
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.x), sizeof(gridDimensions.x));		//int
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.y), sizeof(gridDimensions.y));		//int
		fileStream->write(reinterpret_cast<const char*>(&gridDimensions.z), sizeof(gridDimensions.z));		//int

		for (int i = 0; i < gridDimensions.x; i++) {
			for (int j = 0; j < gridDimensions.y; j++) {
				fileStream->write(reinterpret_cast<const char*>(&m_velocityBuffer(i, j)), sizeof(Scalar) * 2);
			}
		}

		if (fileStream->is_open())
			Logger::get() << " Velocity successfully logged and dumped to file: " << velocityExportName << endl;
	}

	template <>
	void DataExporter<Vector3, Array3D>::dumpVelocity() {
		string velocityExportName("Flow Logs/3D/Velocity/" + m_params.getVelocityFilename() + intToStr(m_numDumpedFrames) + ".log");
		
		dimensions_t gridDimensions = m_pFlowSolver->getGrid()->getDimensions();

		auto_ptr<ofstream> fileStream(new ofstream(velocityExportName.c_str(), ofstream::binary));
		for (int i = 0; i < gridDimensions.x; i++) {
			for (int j = 0; j < gridDimensions.y; j++) {
				for (int k = 0; k < gridDimensions.z; k++) {
					fileStream->write(reinterpret_cast<const char*>(&m_velocityBuffer(i, j, k)), sizeof(Scalar) * 3);
				}
			}
		}

		Logger::get() << " Velocity successfully logged and dumped to file: " << velocityExportName << endl;
	}


	template<>
	void DataExporter<Vector2, Array2D>::dumpThinObject() {
		/*for(unsigned int ithCfg = 0; ithCfg < m_pSimCfgs.size(); ithCfg++) {*/
		//	string thinObjectExportName("Flow Logs/ThinObject/" + m_thinObjectFilename + " " + intToStr(ithCfg) +".log");
		//	Logger::get() << " Dumping thinObject simulation to file:  " << m_thinObjectFilename << endl;
		//	int thinObjectSize = m_pObjectPoints[ithCfg]->size();
		//	auto_ptr<ofstream> fileStream(new ofstream(thinObjectExportName.c_str(), ofstream::binary));
		//	fileStream->write(reinterpret_cast<const char*>(&thinObjectSize), sizeof(thinObjectSize));		//Scalar

		//	Scalar totalTime = (1/(Scalar) m_params.frameRate)*m_objectBufferIndices[ithCfg];
		//	fileStream->write(reinterpret_cast<const char*>(&totalTime), sizeof(totalTime));		//Scalar
		//	fileStream->write(reinterpret_cast<const char*>(&m_objectBufferIndices[ithCfg]), sizeof(m_objectBufferIndices[ithCfg]));	//int

		//	for(int t = 0; t < m_objectBufferIndices[ithCfg]; t++) {
		//		for(int i = 0; i < thinObjectSize; i++) {
		//			fileStream->write(reinterpret_cast<const char*>(&(*m_pObjectBuffers[ithCfg])[getSimulationIndex(i, t, thinObjectSize)]), sizeof(Scalar)*2);
		//		}
		//	}
		//}
		//Logger::get() << " Total velocity dumping: 100 % done!" << endl;
	}
	template<>
	void DataExporter<Vector3, Array3D>::dumpThinObject() {
	}


	template class DataExporter<Vector2, Array2D>;
	template class DataExporter<Vector3, Array3D>;

	#pragma endregion
}