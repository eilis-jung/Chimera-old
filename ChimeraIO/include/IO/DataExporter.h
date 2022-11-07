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


#ifndef __CHIMERA_DATA_EXPORTER_
#define __CHIMERA_DATA_EXPORTER_

#pragma once

#include "ChimeraCore.h"
#include "ChimeraGrids.h"
#include "ChimeraCutCells.h"
#include "ChimeraSolvers.h"
#include "ChimeraSolids.h"
#include "ChimeraRendering.h"

namespace Chimera {
	using namespace CutCells;
	using namespace Solvers;

	template <class VectorType, template <class> class ArrayType>
	class DataExporter {
	public:

		#pragma region PublicStructures
		typedef struct configParams_t {
			/** Logging variables */
			bool logVelocity;
			bool logDensity;
			bool logPressure;
			bool logThinObject;
			bool logSpecialCells;
			bool logScreenshot;

			string m_densityFilename;
			string m_pressureFilename;
			string m_velocityFilename;
			string m_positionFilename;
			string m_thinObjectFilename;
			string m_specialCellsFilename;
			string m_screenshotFilename;

			int m_screenWidth;
			int m_screenHeight;

			const string &getDensityFilename() const {
				return m_densityFilename;
			}

			void setDensityFilename(const string &densityFilename) {
				m_densityFilename = densityFilename;
			}

			const string &getPressureFilename() const {
				return m_pressureFilename;
			}
			void setPressureFilename(const string &pressureFilename) {
				m_pressureFilename = pressureFilename;
				Logger::getInstance()->get() << "DataExporter: logging pressure field into " << m_pressureFilename << endl;
			}

			const string &getVelocityFilename() const {
				return m_velocityFilename;
			}
			void setVelocityFilename(const string &velocityFilename) {
				m_velocityFilename = velocityFilename;
				Logger::getInstance()->get() << "DataExporter: logging velocity field into " << m_velocityFilename << endl;
			}

			const string &getCutCellsFilename() const {
				return m_specialCellsFilename;
			}
			void setCutCellsFilename(const string &cutCellsFilename) {
				m_specialCellsFilename = cutCellsFilename;
				Logger::getInstance()->get() << "DataExporter: logging cut-cells into " << m_specialCellsFilename << endl;
			}

			const string &getScreenshotFilename() const {
				return m_screenshotFilename;
			}
			void setScreenshotFilename(const string &screenshotFilename) {
				m_screenshotFilename = screenshotFilename;
				Logger::getInstance()->get() << "DataExporter: logging screenshot into " << m_screenshotFilename << endl;
			}

			int getScreenWidth() const {
				return m_screenWidth;
			}

			int getScreenHeight() const {
				return m_screenHeight;
			}

			void setScreenSize(int w, int h) {
				m_screenWidth = w;
				m_screenHeight = h;
			}

			/** Logging frame rate*/
			int frameRate;
			Scalar totalSimulatedTime;

			CutCellsBase<VectorType> *pSpecialCells;
			ArrayType<VectorType> *pNodeBasedVelocities;

			configParams_t() {
				logVelocity = logDensity = logPressure = logThinObject = logSpecialCells = logScreenshot = false;
				m_screenWidth = m_screenHeight = 0;
				frameRate = 30;
				totalSimulatedTime = 0.0f;
				pSpecialCells = NULL;
				pNodeBasedVelocities = NULL;
			}
		} configParams_t;
		#pragma endregion

	private:

		#pragma region ClassMembers
		configParams_t m_params;
		FlowSolver<VectorType, ArrayType> *m_pFlowSolver;

		ArrayType<Scalar> m_densityFields;
		ArrayType<Scalar> m_pressureFields;
		ArrayType<VectorType> m_velocityBuffer;

		vector<vector<VectorType>> m_objectPoints;
		vector<vector<VectorType>> m_objectBuffers;

		vector<unsigned char> m_pixels;
		SDL_Surface *m_sdlSurface;

		int m_numDumpedFrames;
		#pragma endregion

		#pragma region PrivateFunctionalities
		FORCE_INLINE int getSimulationIndex(int i, int j, int t, const dimensions_t &gridDim) {
			return t*gridDim.x*gridDim.y +
				j*gridDim.x +
				i;
		}

		FORCE_INLINE int getSimulationIndex(int i, int t, int bufferSize) {
			return t*bufferSize + i;
		}
		#pragma endregion

		#pragma region LoggingFunctions
		void logDumpDensity(Scalar timeElapsed);
		void logDumpPressure(Scalar timeElapsed);
		void logDumpVelocity(Scalar timeElapsed);
		void logDumpThinObject(Scalar timeElapsed);
		void logDumpSpecialCells(Scalar timeElapsed);
		void logSaveScreenshot();
		#pragma endregion

		#pragma region DumpingFunctions
		void dumpDensity();
		void dumpPressure();
		void dumpVelocity();
		void dumpThinObject();
		#pragma endregion

	public:
		
		#pragma region Constructors
		DataExporter(const configParams_t &params, const dimensions_t &gridDimensions);
		#pragma endregion

		#pragma region Destructor
		virtual ~DataExporter() { SDL_FreeSurface(m_sdlSurface);  }
		#pragma endregion

		#pragma region AccessFunctions
		FORCE_INLINE void setFlowSolver(FlowSolver<VectorType, ArrayType> *pFlowSolver) {
			m_pFlowSolver = pFlowSolver;
		}

		void addObjectPoints(vector<VectorType> *pObjectPoints) {
			m_objectPoints.push_back(*pObjectPoints);
		}

		configParams_t & getParams() {
			return m_params;
		}
		#pragma endregion

		#pragma region Functionalities
		FORCE_INLINE void log(Scalar timeElapsed) {
			if (timeElapsed >= m_numDumpedFrames*(1 / (float)m_params.frameRate)) {
				if (m_params.logDensity) {
					logDumpDensity(timeElapsed);
				}
				if (m_params.logPressure) {
					logDumpPressure(timeElapsed);
				}
				if (m_params.logVelocity) {
					logDumpVelocity(timeElapsed);
				}
				if (m_params.logThinObject) {
					logDumpThinObject(timeElapsed);
				}

				if (m_params.logSpecialCells) {
					logDumpSpecialCells(timeElapsed);
				}

				if (m_params.logScreenshot) {
					logSaveScreenshot();
				}
				++m_numDumpedFrames;
			}
		}
		#pragma endregion
	};
}

#endif