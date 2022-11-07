//This class is used to export vector fields of the grid. The Exporter
//can help to load vector fields into maya fluid container. Displaying the
//velocity draw component in fluid container can show the vector field explicitly.

#ifndef __CHIMERA_VEL_FIELD_EXPORTER_
#define __CHIMERA_VEL_FIELD_EXPORTER_

#pragma once

#include "ChimeraRendering.h"
#include "ChimeraCore.h"


namespace Chimera {
	using namespace Core;

	class VelFieldExporter {
	public:
#pragma region Constructors
		VelFieldExporter(GridData3D* grid_data_d3, Scalar endtime, Scalar samplingrate, string path);
		~VelFieldExporter() {
		}
#pragma endregion

		void dumpFrame();
	private:
		void dumpXML();
		GridData3D* m_pGridData3D;
		DoubleBuffer<Scalar, Array3D> m_DensityBuffer;
		Scalar m_endtime;
		Scalar m_samplingrate;
		string m_path;
		unsigned int m_currFrame;
	};
}

#endif
