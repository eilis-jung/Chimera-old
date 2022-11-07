// ===================================================================================
// exports a vector field, stored in z-fastest order to an Amira file.
// dim: resolution of the grid on which the data is given
// bounds: xmin, xmax, ymin, ymax, zmin, zmax

#ifndef __CHIMERA_AMIRA_EXPORTER_
#define __CHIMERA_AMIRA_EXPORTER_

#pragma once
#include <string.h>
#include <vector>
#include "ChimeraRendering.h"


namespace Chimera {
	class AmiraExporter {
	public:
#pragma region Constructors
		AmiraExporter(GridData3D* grid_data_d3, std::string path, int dim[3], double bounds[6]);
		~AmiraExporter() {
		}
#pragma endregion
		void exportVector();		
	private:		
		bool ExportVectorFieldToAmira(const std::string& exportPath, const std::vector<float>& vectorData, int dim[3], double bounds[6]);
		GridData3D* m_pGridData3D;
		string m_path;
		std::vector<float> m_vectorData;
		int m_dim[3];
		double m_bounds[6];
		unsigned int m_currFrame;
	};
}

#endif