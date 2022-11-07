#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <algorithm>
#include <assert.h>
#include <vector>
#include "IO/AmiraExporter.h"

namespace Chimera {

#pragma region Constructors
	AmiraExporter::AmiraExporter(GridData3D* grid_data3, string path, int dim[3], double bounds[6]):
		m_pGridData3D(grid_data3)
	{
		for(int i=0; i<3; i++)
		{
			m_dim[i] = dim[i];
		}
		for (int i = 0; i<6; i++)
		{
			m_bounds[i] = bounds[i];
		}
		m_currFrame = 1;
		m_path = path;
	}
	bool AmiraExporter::ExportVectorFieldToAmira(const std::string& exportPath, const std::vector<float>& vectorData, int dim[3], double bounds[6]) 
	{
		// Write header
		{
			std::ofstream outStream(exportPath);
			outStream << "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1" << std::endl;
			outStream << std::endl;
			outStream << std::endl;

			outStream << "define Lattice " << dim[0] << " " << dim[1] << " " << dim[2] << std::endl;
			outStream << std::endl;

			outStream << "Parameters {" << std::endl;
			outStream << "Content \"" << dim[0] << "x" << dim[1] << "x" << dim[2] << "	float[3], uniform coordinates\"," << std::endl;

			outStream << "\tBoundingBox " << bounds[0] << " " << bounds[1] << " " << bounds[2] << " " << bounds[3] << " " << bounds[4] << " " << bounds[5] << "," << std::endl;
			outStream << "\tCoordType \"uniform\"" << std::endl;
			outStream << "}" << std::endl;
			outStream << std::endl;

			outStream << "Lattice { float[3] Data } @1" << std::endl;
			outStream << std::endl;

			outStream << "# Data section follows" << std::endl;
			outStream << "@1" << std::endl;

			outStream.close();
		}

		// Write data
		{
			std::string dirPath = exportPath.substr(0, max(int(exportPath.find_last_of("\\")), int(exportPath.find_last_of("/"))) + 1);
			CreateDirectoryA(dirPath.c_str(), NULL);

			std::ofstream outStream(exportPath, std::ios::out | std::ios::app | std::ios::binary);
			outStream.write((char*)vectorData.data(), sizeof(float) * vectorData.size());
			outStream.close();
		}
		return true;
	}

	void AmiraExporter::exportVector()
	{
		if (m_currFrame == 1 || m_currFrame == 180)
		{
			m_currFrame = 1;
		}
		string framExportName(m_path + "Frame" + intToStr(m_currFrame) + ".am");
		size_t resolution_x = m_pGridData3D->getDimensions().x;
		size_t resolution_y = m_pGridData3D->getDimensions().y;
		size_t resolution_z = m_pGridData3D->getDimensions().z;
		m_vectorData.clear();
		for (int k = 0; k < resolution_z; k++)
		{
			for (int j = 0; j < resolution_y; j++)
			{
				for (int i = 0; i < resolution_x; i++)
				{
					m_vectorData.push_back(m_pGridData3D->getVelocity(i, j, k).x);
					m_vectorData.push_back(m_pGridData3D->getVelocity(i, j, k).y);
					m_vectorData.push_back(m_pGridData3D->getVelocity(i, j, k).z);
				}
			}
		}
		m_currFrame++;
		ExportVectorFieldToAmira(framExportName, m_vectorData, m_dim, m_bounds);
	}
}