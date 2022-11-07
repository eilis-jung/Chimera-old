#include "MayaCache/VelFieldExporter.h"
#include "MayaCache/XMLWriter.h"
#include "MayaCache/MCXMemoryWriter.h"

namespace Chimera {

#pragma region Constructors
	VelFieldExporter::VelFieldExporter(GridData3D* grid_data3, Scalar endtime, Scalar samplingrate, string path) :
		m_pGridData3D(grid_data3),
		m_endtime(endtime),
		m_samplingrate(samplingrate),
		m_DensityBuffer(m_pGridData3D->getDimensions()),
		m_path(path)
	{
		m_currFrame = 1;
		m_DensityBuffer = m_pGridData3D->getDensityBuffer();
		//m_pVelocity = m_pGridData3D->getVelocityArrayPtr();
	}
#pragma region DumpingFunctions
	void VelFieldExporter::dumpXML() {

		string xmlExportName(m_path + intToStr(1) + ".xml");
		//std::string xml_filename("D:\\nCacheWriter\\nCacheWriter\\Debug\\test.xml");
		nCache::XMLWriter xml_writer;
		xml_writer.init("OneFilePerFrame", "mcx", 0, m_endtime, m_samplingrate, "2.0");

		nCache::ChannelInfo channel_info;
		channel_info._channel_name = "fluidShape1_density";
		channel_info._channel_interpretation = "density";
		channel_info._sampling_type = nCache::ChannelInfo::SamplingType::REGULAR;
		channel_info._channel_type = "FloatArray";
		channel_info._sampling_rate = m_samplingrate;
		channel_info._start_time = 250;
		channel_info._end_time = m_endtime;
		xml_writer.addChannel("channel0", channel_info);

		channel_info._channel_name = "fluidShape1_velocity";
		channel_info._channel_interpretation = "velocity";
		channel_info._sampling_type = nCache::ChannelInfo::SamplingType::REGULAR;
		channel_info._channel_type = "FloatArray";
		channel_info._sampling_rate = m_samplingrate;
		channel_info._start_time = 250;
		channel_info._end_time = m_endtime;
		xml_writer.addChannel("channel1", channel_info);

		channel_info._channel_name = "fluidShape1_resolution";
		channel_info._channel_interpretation = "resolution";
		channel_info._sampling_type = nCache::ChannelInfo::SamplingType::REGULAR;
		channel_info._channel_type = "FloatArray";
		channel_info._sampling_rate = m_samplingrate;
		channel_info._start_time = 250;
		channel_info._end_time = m_endtime;
		xml_writer.addChannel("channel2", channel_info);

		channel_info._channel_name = "fluidShape1_offset";
		channel_info._channel_interpretation = "offset";
		channel_info._sampling_type = nCache::ChannelInfo::SamplingType::REGULAR;
		channel_info._channel_type = "FloatArray";
		channel_info._sampling_rate = m_samplingrate;
		channel_info._start_time = 250;
		channel_info._end_time = m_endtime;
		xml_writer.addChannel("channel3", channel_info);

		//initialize writer
		xml_writer.write(xmlExportName);
	}

	void VelFieldExporter::dumpFrame()
	{
		if (m_currFrame == 1 || m_currFrame == 180)
		{
			dumpXML();
			m_currFrame = 1;
		}
			
		nCache::Header o_header;
		size_t stored_density, stored_velocity;
		size_t resolution_x = m_pGridData3D->getDimensions().x;
		size_t resolution_y = m_pGridData3D->getDimensions().y;
		size_t resolution_z = m_pGridData3D->getDimensions().z;
		size_t grid_size = resolution_x*resolution_y*resolution_z;
		//size_t density_size = resolution_x*resolution_y*resolution_z;
		size_t velocity_size = (resolution_x + 1)*resolution_y*resolution_z
			+ (resolution_y + 1)*resolution_x*resolution_z
			+ (resolution_z + 1)*resolution_x*resolution_y;
		if (grid_size % 2 != 0) stored_density = grid_size + 1;
		else stored_density = grid_size;
		if (velocity_size % 2 != 0) stored_velocity = velocity_size + 1;
		else stored_velocity = velocity_size;
		o_header.header_blob_size = 76;
		//  |---MYCH (Group)			// 4
		//  |     |---CHNM				// 4
		//  |     |---Dummy	number		// 8
		//  |     |---Channel name size	// 4
		//  |     |---Channel name		// flexible(24)
		//  |     |---SIZE				// 4
		//  |     |---Dummy	number		// 8
		//  |     |---Dummy	number		// 4
		//  |     |---Channel data size	// 4
		//  |     |---Dummy	number		// 4
		//  |     |---Channel data type	// 4
		//  |     |---Dummy	number		// 8
		//  |     |---Buffer size		// 4
		//  |     |---DVCA				// flexible (Double Vector Array)
		//  |     |..
		o_header.channels_blob_size = 4 + 4 * (56 + 24) + 16 + 16 + stored_density * 4 + stored_velocity * 4;
		o_header.VRSN = "0.1";
		o_header.STIM = m_samplingrate * m_currFrame;
		o_header.ETIM = m_samplingrate * m_currFrame;
		string framExportName(m_path + intToStr(1) + "Frame" + intToStr(m_currFrame) + ".mcx");

		nCache::MCXMemoryWriter mcx_writer(framExportName, o_header);
		mcx_writer.write_header(o_header);

		nCache::ChannelDataContainer channels_data;
		nCache::ChannelData o_density, o_velocity, o_resolution, o_offset;
		std::vector<float> test_velocity;

		o_density._type = nCache::ChannelDataType::FBCA;
		o_velocity._type = nCache::ChannelDataType::FBCA;
		o_resolution._type = nCache::ChannelDataType::FBCA;
		o_offset._type = nCache::ChannelDataType::FBCA;

		o_density._fbca.resize(stored_density);
		o_density._real_size = grid_size;

		test_velocity.resize(3 * grid_size);
		o_velocity._fbca.resize(stored_velocity);
		o_velocity._real_size = velocity_size;

		o_resolution._fbca.resize(4);
		o_resolution._real_size = 3;

		o_offset._fbca.resize(4);
		o_offset._real_size = 3;

		o_resolution._fbca[0] = resolution_x;
		o_resolution._fbca[1] = resolution_y;
		o_resolution._fbca[2] = resolution_z;

		o_offset._fbca[0] = 0;
		o_offset._fbca[1] = 0;
		o_offset._fbca[2] = 0;

		int index;
		for (int k = 0; k < resolution_z; k++) 
		{
			for (int j = 0; j < resolution_y; j++) 
			{
				for (int i = 0; i < resolution_x; i++) 
				{
					index = k*resolution_x*resolution_y + j*resolution_x + i;
					//The way to access density needs to be modfied
					o_density._fbca[index] = m_pGridData3D->getVorticity(i, j, k);
					test_velocity[index] = m_pGridData3D->getVelocity(i, j, k).x;
					test_velocity[grid_size + index] = m_pGridData3D->getVelocity(i, j, k).y;
					test_velocity[2 * grid_size + index] = m_pGridData3D->getVelocity(i, j, k).z;
				}
			}
		}
		//x component
		int reformedvec_index_x = 0;
		for (int i = 0; i<grid_size; )
		{
			if (i == 0 || (reformedvec_index_x + 1) % (resolution_x + 1) != 0)
			{
				o_velocity._fbca[reformedvec_index_x] = test_velocity[i];
				reformedvec_index_x++;
				i++;
			}
			else
			{
				o_velocity._fbca[reformedvec_index_x] = 0;
				reformedvec_index_x++;
			}
		}
		reformedvec_index_x++;
		//y component
		int reformedvec_index_y = 0;
		for (int i = grid_size; i < 2 * grid_size;)
		{
			for (int j = 0; j<resolution_z; j++)
			{
				for (int k = 0; k<resolution_x*resolution_y; k++)
				{
					o_velocity._fbca[reformedvec_index_x + reformedvec_index_y] = test_velocity[i];
					reformedvec_index_y++;
					i++;
				}
				for (int k = 0; k<resolution_x; k++)
				{
					o_velocity._fbca[reformedvec_index_x + reformedvec_index_y] = 0;
					reformedvec_index_y++;
				}
			}
		}
		//z component
		int reformedvec_index_z = 0;
		for (int i = 2 * grid_size; i < 3 * grid_size; i++)
		{
			o_velocity._fbca[reformedvec_index_x + reformedvec_index_y + reformedvec_index_z] = test_velocity[i];
			reformedvec_index_z++;
		}
		//velocity
		channels_data.insert(nCache::ChannelDataContainer::value_type("fluidShape1_density", o_density));
		channels_data.insert(nCache::ChannelDataContainer::value_type("fluidShape1_velocity", o_velocity));
		channels_data.insert(nCache::ChannelDataContainer::value_type("fluidShape1_resolution", o_resolution));
		channels_data.insert(nCache::ChannelDataContainer::value_type("fluidShape1_offset", o_offset));

		mcx_writer.write_channel(channels_data);
		mcx_writer.write(o_header);


		m_currFrame++;
	}

#pragma endregion
}