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


#ifndef __CHIMERA_DATA_EXPORTER_3D_H_
#define __CHIMERA_DATA_EXPORTER_3D_H_

#pragma once

/************************************************************************/
/* Core                                                                 */
/************************************************************************/
#include "ChimeraCore.h"

#include "ChimeraGrids.h"
#include "ChimeraCutCells.h"
#include "ChimeraSolvers.h"
#include "ChimeraSolids.h"

namespace Chimera {

	namespace Data {

		template <class VectorT>
		class DataExporter {
		public:
			/************************************************************************/
			/* Public structures                                                    */
			/************************************************************************/
			typedef struct configParams_t {
				/** Logging variables */
				bool logVelocity;
				bool logDensity;
				bool logPressure;
				bool logThinObject;
				bool logSpecialCells;
				bool logStreamfunctions;
				bool logScreenshot;

				/** Logging frame rate*/
				int frameRate;
				Scalar totalSimulatedTime;


				configParams_t() {
					logVelocity = logDensity = logPressure = logThinObject = logSpecialCells = logStreamfunctions = false;
					frameRate = 30;
					totalSimulatedTime = 0.0f;
					pSpecialCells = NULL;
					pSpecialCells3D = NULL;
					pNodeVelocityField3D = NULL;
					cutCellInterpolationMethod = mvcInterpolation;
				}
			} configParams_t;

		private:

			/************************************************************************/
			/* Class members                                                        */
			/************************************************************************/
			configParams_t m_params;
			vector<SimulationConfig<VectorT> *> m_pSimCfgs;

			Array2D<Scalar> m_densityFields2D;
			Array3D<Scalar> m_densityFields3D;

			Array2D<Scalar> m_pressureFields2D;
			Array3D<Scalar> m_pressureFields3D;

			Array2D<VectorT> m_velocityBuffer2D;
			Array3D<VectorT> m_velocityBuffer3D;

			vector<vector<VectorT>> m_objectPoints;
			vector<vector<VectorT>> m_objectBuffers;

			int m_numDumpedFrames;

			string m_densityFilename;
			string m_pressureFilename;
			string m_velocityFilename;
			string m_positionFilename;
			string m_thinObjectFilename;
			string m_specialCellsFilename;
			string m_streamfunctionFilename;

			/************************************************************************/
			/* Buffer utils                                                         */
			/************************************************************************/
			void allocateBuffers(SimulationConfig<VectorT> *pSimCfg);

			FORCE_INLINE int getSimulationIndex(int i, int j, int t, const dimensions_t &gridDim) {
				return t*gridDim.x*gridDim.y +
					j*gridDim.x +
					i;
			}

			FORCE_INLINE int getSimulationIndex(int i, int t, int bufferSize) {
				return t*bufferSize + i;
			}

			FORCE_INLINE int getSimulationIndex(int i, int j, int k, int t, const dimensions_t &gridDim) {
				return	t*gridDim.x*gridDim.y*gridDim.z +
					k*gridDim.x*gridDim.y +
					j*gridDim.x +
					i;
			}

			/************************************************************************/
			/* Logging functions                                                    */
			/************************************************************************/
			void logDumpDensity(Scalar timeElapsed);
			void logDumpPressure(Scalar timeElapsed);
			void logDumpVelocity(Scalar timeElapsed);
			void logDumpThinObject(Scalar timeElapsed);
			void logDumpSpecialCells(Scalar timeElapsed);
			void logDumpStreamfunction(Scalar timeElapsed);

			/************************************************************************/
			/* Dumping functions                                                    */
			/************************************************************************/
			void dumpDensity();
			void dumpPressure();
			void dumpVelocity();
			void dumpPosition();
			void dumpThinObject();
			void dumpSpecialCells();
			void dumpPolygonMeshes();
			void dumpMeshNodeVelocities();

		public:
			/************************************************************************/
			/* Ctors                                                                */
			/************************************************************************/
			DataExporter(const configParams_t &params, const dimensions_t &gridDimensions);
			/************************************************************************/
			/* Access functions                                                     */
			/************************************************************************/
			//Within each added simulation config, memory used for logging will be allocated
			void addSimulationConfig(SimulationConfig<VectorT> *pSimCfg) {
				m_pSimCfgs.push_back(pSimCfg);
			}

			void setDensityFilename(const string &densityFilename) {
				m_densityFilename = densityFilename;
			}

			void setPressureFilename(const string &pressureFilename) {
				m_pressureFilename = pressureFilename;
			}

			void setVelocityFilename(const string &velocityFilename) {
				m_velocityFilename = velocityFilename;
				Logger::getInstance()->get() << "DataExporter: logging velocity field" << endl;
			}

			void setCutCellsFilename(const string &cutCellsFilename) {
				m_specialCellsFilename = cutCellsFilename;
				Logger::getInstance()->get() << "DataExporter: logging cut-cells" << endl;
			}

			void setStreamfunctionFilename(const string &streamfunctionFilename) {
				m_streamfunctionFilename = streamfunctionFilename;
				Logger::getInstance()->get() << "DataExporter: logging streamfunctions" << endl;
			}

			void addObjectPoints(vector<Vector2> *pObjectPoints) {
				m_pObjectPoints.push_back(pObjectPoints);
			}

			configParams_t & getParams() {
				return m_params;
			}
			/************************************************************************/
			/* Functionalities                                                      */
			/************************************************************************/
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

					if (m_params.logStreamfunctions) {
						logDumpStreamfunction(timeElapsed);
					}
					++m_numDumpedFrames;
				}
			}
		};



	}

}

#endif